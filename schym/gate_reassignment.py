import gym
import numpy as np
from gym import spaces


class Task:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time

    @property
    def duration(self):
        return self.end_time - self.start_time

    def __repr__(self):
        return f"Task(start_time={self.start_time}, end_time={self.end_time})"


def generate_task(start_time, avg_duration):
    duration = np.random.poisson(lam=avg_duration - 1) + 1
    end_time = start_time + duration
    return Task(start_time=start_time, end_time=end_time)


def generate_single_resource_schedule(length, avg_spread, avg_duration):
    tasks = []

    start_time = np.random.poisson(avg_spread)
    while start_time < length:
        task = generate_task(start_time, avg_duration)
        if task.end_time < length:
            tasks.append(task)
            start_time = task.end_time + np.random.poisson(lam=avg_spread)
        else:
            break
    return tasks


def generate_random_schedule(length, n_resources, avg_spread, avg_duration):
    return [
        generate_single_resource_schedule(length, avg_spread=avg_spread, avg_duration=avg_duration)
        for _ in range(n_resources)
    ]


def intersects(task1, task2):
    return (
        task1.start_time < task2.end_time <= task1.end_time
    ) or (
        task2.start_time < task1.end_time <= task2.end_time
    ) or (
        task1.start_time >= task2.start_time and
        task1.end_time <= task2.end_time
    ) or (
        task2.start_time >= task1.start_time and
        task2.end_time <= task1.end_time
    )


class GateScheduling(gym.Env):
    # Represent each scheduled gate using more than a single pixel
    # this allows for more granularity and also to indicate flights that
    # are planned back to back
    BAR_PIXELS = 5

    def __init__(
            self,
            n_resources=10,
            max_len=150,
            visible_window=30,
            avg_spread=2,
            avg_duration=3
    ):
        self.n_resources = n_resources
        self.max_len = max_len
        self.visible_window = visible_window
        self.avg_spread = avg_spread
        self.avg_duration = avg_duration
        self.schedule = None

        self.action_space = spaces.Discrete(self.n_resources + 2)
        self.observation_space = spaces.Box(
            low=0.,
            high=1.,
            dtype=np.float32,
            shape=((self.n_resources + 1) * GateScheduling.BAR_PIXELS,
                   self.visible_window * GateScheduling.BAR_PIXELS,
                   3)
        )

        self.current_timestep = 0
        self.tasks_to_reschedule = []

        self.viewer = None

        self.reset()

    def observe(self):
        h, w = GateScheduling.BAR_PIXELS, GateScheduling.BAR_PIXELS
        schedule_repr = np.zeros(
            ((self.n_resources + 1) * h, (self.max_len * w), 3),
            dtype=np.float32
        )

        # Create a basic gantt chart of all the tasks scheduled
        for i, tasks in enumerate(self.schedule):
            for j, task in enumerate(tasks):
                schedule_repr[i * h + 1: (i+1) * h - 1,
                              task.start_time * w + 1:task.end_time * w - 1,
                              1] = 1

        if len(self.tasks_to_reschedule) > 0:
            # Add the task to be rescheduled as a different color on the bottom row
            current_task_to_reschedule = self.tasks_to_reschedule[0]

            ttr_start_time = current_task_to_reschedule.start_time
            ttr_end_time = current_task_to_reschedule.end_time
            schedule_repr[self.n_resources * h + 1:(self.n_resources + 1) * h - 1,
                          ttr_start_time * w + 1:ttr_end_time * w - 1,
                          0] = 1

        visible_from = self.current_timestep * w
        visible_to = (self.current_timestep + self.visible_window) * w
        return schedule_repr[:, visible_from:visible_to, ...]

    def reset(self):
        self.schedule = [
            generate_single_resource_schedule(
                self.max_len,
                avg_spread=self.avg_spread,
                avg_duration=self.avg_duration
            )
            for _ in range(self.n_resources)
        ]

        # Generate a random task to reschedule in the initial visible window. If it would not
        # immediately visible the problem would make no sense.
        task_to_reschedule = generate_task(
            start_time=np.random.randint(0, self.visible_window),
            avg_duration=self.avg_duration
        )
        while task_to_reschedule.end_time > self.max_len:
            task_to_reschedule = generate_task(
                start_time=np.random.randint(0, self.max_len),
                avg_duration=self.avg_duration
            )

        self.current_timestep = 0
        self.tasks_to_reschedule = [
            task_to_reschedule
        ]

        return self.observe()

    def step_to_next_timestep(self):
        self.current_timestep += 1

    def end_of_time(self):
        return self.current_timestep + self.visible_window == self.max_len

    def in_time_to_reschedule_remaining_tasks(self):
        """Check whether all tasks can still be rescheduled in time"""
        return all(
            map(lambda task: task.start_time >= self.current_timestep, self.tasks_to_reschedule)
        )

    def has_tasks_to_reschedule(self):
        """Check if there are still tasks to reschedule."""
        return len(self.tasks_to_reschedule) > 0

    def action_is_moving_flight_in_time(self, action):
        """Check if the action is about moving a flight in time"""
        return action in [self.n_resources, self.n_resources + 1]

    def step(self, action):
        if not self.has_tasks_to_reschedule():
            return self.reset()

        if not self.in_time_to_reschedule_remaining_tasks():
            reward = -50
            return self.observe(), reward, True, {}

        if self.end_of_time():
            return self.observe(), 0, True, {}

        task_to_schedule = self.tasks_to_reschedule.pop(0)

        if not self.action_is_moving_flight_in_time(action):
            tasks_on_resource = self.schedule[action]

            # All tasks that are already on this resource can remain there as long as they do not
            # overlap with the newly added task. Also the newly added task will be added there.
            new_tasks_on_resource = (
                    [task for task in tasks_on_resource if not intersects(task, task_to_schedule)] +
                    [task_to_schedule]
            )

            # Some tasks might need to be rescheduled again since they have a new conflict by
            # this action. We add those to the list of tasks that need to be rescheduled.
            new_tasks_to_rescheulde = [
                task for task in tasks_on_resource if intersects(task, task_to_schedule)
            ]

            cost_of_tasks_to_reschedule = sum(
                map(lambda t: t.end_time - t.start_time, new_tasks_to_rescheulde)
            )
            self.tasks_to_reschedule.extend(new_tasks_to_rescheulde)
            self.schedule[action] = new_tasks_on_resource

            # We move the simulation one timestep before checking if we still have a
            # feasible situation
            self.step_to_next_timestep()

            if not self.in_time_to_reschedule_remaining_tasks():
                # In case we are no longer in time to reschedule all tasks in the new situation
                # the simulation ends
                return self.observe(), -50, True, {}
            elif self.has_tasks_to_reschedule():
                # If there are still tasks that haven't been settled yet, proceed. The penalty
                # in this case is given by the number of tasks that need to be rescheduled extra
                # because of this
                return self.observe(), -1 * len(new_tasks_to_rescheulde), False, {}
            else:
                # Otherwise, we are done rescheduling
                return self.observe(), 100, True, {}

        elif self.action_is_moving_flight_in_time(action):
            self.tasks_to_reschedule = [task_to_schedule] + self.tasks_to_reschedule

            if ((action == self.n_resources) and
                (task_to_schedule.start_time > self.current_timestep)):
                # The task can only be moved forward is there still is time and moving the task
                # does not cause a situation where we are too late to reschedule this task
                task_to_schedule.start_time = task_to_schedule.start_time - 1
                task_to_schedule.end_time = task_to_schedule.end_time - 1

                self.step_to_next_timestep()
                return self.observe(), -5, False, {}

            elif ((action == self.n_resources + 1) and
                  (task_to_schedule.end_time < self.max_len)):
                task_to_schedule.start_time = task_to_schedule.start_time + 1
                task_to_schedule.end_time = task_to_schedule.end_time + 1

                self.step_to_next_timestep()
                return self.observe(), -5, False, {}

            self.step_to_next_timestep()
            if not self.in_time_to_reschedule_remaining_tasks():
                # In case we are no longer in time to reschedule all tasks in the new situation
                # the simulation ends
                return self.observe(), -50, True, {}
            else:
                # In this case a non-permissible action has been performed with regards to moving
                # in time
                return self.observe(), -2, False, {}
