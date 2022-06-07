import os, sys
commit_id = 'b5885615eeddfc249758d003a99c6854884a94b9'
pytimeloop_dir = "../pytimeloop"
working_path = os.getcwd()
dst_path = os.path.join(working_path, pytimeloop_dir)
try:
    os.system("git clone https://github.com/Accelergy-Project/timeloop-python.git {}".format(dst_path))
    os.chdir(f'{dst_path}')
    os.system(f"git checkout {commit_id}")
    os.system(f'git am {working_path}/patches/update_interface.patch')
    os.system('git submodule update --init')
    os.system('rm -rf build')
    os.system('pip install -e .')
except:
    "Something wring when installing pytimeloop, please check pytimeloop repository for detailed installation step"


