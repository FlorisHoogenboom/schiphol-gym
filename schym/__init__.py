from gym.envs import registration

registration.register(
    id='gate-reassignment-v1',
    entry_point='schym.gate_reassignment:GateScheduling'
)