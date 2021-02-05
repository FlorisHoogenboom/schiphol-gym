from setuptools import setup, find_packages

setup(
    name='schiphol-gym',
    version='0.0.1',
    packages=find_packages(),
    url='',
    license='',
    author='Floris Hoogenboom',
    author_email='floris@digitaldreamworks.nl',
    description='Gym environments for a variety of Airport Processes',
    install_requires=['gym>=0.17.3'],
    extras_require={
        'experiments': [
            'tf-agents[reverb]==0.7.1',
            'tensorboard==2.4.1',
            'tensorflow==2.4.1'
        ]
    }
)
