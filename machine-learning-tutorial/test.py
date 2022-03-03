import matplotlib
print(matplotlib.rcParams['backend'])
print(matplotlib.rcParams['interactive'])
print(matplotlib.matplotlib_fname())
import matplotlib.rcsetup as rcsetup
print(rcsetup.all_backends)

