from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    hypen_e_dot = "-e ."
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [r.replace("\n","") for r in requirements]
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    return requirements

setup (
name = "mlproject",
version = '0.0.1' ,
author = 'Rohan' ,
author_email = 'rohankarabhimgol@gmail.com' ,
packages = find_packages() ,
install_requires = get_requirements('requirements.txt')
)
