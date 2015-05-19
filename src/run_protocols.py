import subprocess
import os


if __name__ == '__main__':
    script = ['protocolA.py', 'protocolB.py', 'protocolC.py', 'protocolD.py',
              'protocolE.py', 'protocolE2.py']
    experiment = ['protocolA', 'protocolB', 'protocolC', 'protocolD',
                  'protocolE', 'protocolE2']
    path = '../params/'
    pfname = ['params_protocolA.cfg', 'params_protocolB.cfg',
              'params_protocolC.cfg', 'params_protocolD.cfg',
              'params_protocolE.cfg', 'params_protocolE2.cfg']

    # script = ['protocolEfficiency.py']
    # experiment = ['protocolEfficiency']
    # pfname = ['params_protocolA.cfg']

    # script = ['protocolDelays.py']
    # experiment = ['protocolDelays']
    # pfname = ['params_protocolA.cfg']

    # script = ['protocolVelocity.py']
    # experiment = ['protocolVelocity']
    # pfname = ['params_protocolA.cfg']

    # script = ['brute_force.py']
    # experiment = ['bruteForce']
    # pfname = ['params_protocolA.cfg']

    for i in range(len(experiment)):
        print '----------------------------------------------------------'
        print 'Running {}!'.format(experiment[i])
        subprocess.call(["python", script[i], path+pfname[i], experiment[i]])
        print '----------------------------------------------------------'

    os.system('beep -f 200 -r 2 -d 90 -l 400')
