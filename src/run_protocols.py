import subprocess
import os


if __name__ == '__main__':
    script = ['protocolA.py', 'protocolB.py', 'protocolC.py', 'protocolD.py',
              'protocolD2.py']
    experiment = ['protocolA', 'protocolB', 'protocolC', 'protocolD',
                  'protocolD2']
    path = '../params/'
    pfname = ['params_protocolA.cfg', 'params_protocolB.cfg',
              'params_protocolC.cfg', 'params_protocolD.cfg',
              'params_protocolD2.cfg']

    for i in range(len(experiment)):
        print '----------------------------------------------------------'
        print 'Running {}!'.format(experiment[i])
        subprocess.call(["python", script[i], path+pfname[i], experiment[i]])
        print '----------------------------------------------------------'

    os.system('beep -f 200 -r 2 -d 90 -l 400')
