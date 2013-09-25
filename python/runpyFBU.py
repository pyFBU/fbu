#! /bin/env python
###################
# davide.gerbaudo@cern.ch clement.helsens@cern.ch, francesco.rubbo@cern.ch
###################
# usage:
# python runpyFBU.py 
###################

import os
from pyFBU import pyFBU

#__________________________________________________________
if __name__ == "__main__":
    projectDir = os.path.dirname(os.path.abspath(__file__)).replace('/python','')
    dataDir = projectDir+'/data/'
    defaultData = dataDir+'data.json'
    defaultMig = dataDir+'migrations.json'
    defaultBkg = dataDir+'background.json'
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option ('-D', '--data', default=defaultData, help="json data file")
    parser.add_option ('-M', '--matrix', default=defaultMig, help="json migration matrix file")
    parser.add_option ('-B', '--background', default=defaultBkg, help="json background file")
    parser.add_option ('-t', '--template', help='input model template file')
    parser.add_option ('-m', '--model', help='name of the model. Models are automatically created under the same directory as the template')
    parser.add_option ('-v', '--verbose', help='Toggle verbose', action='store_true', default=False)
    (opts, args) = parser.parse_args()
    if not opts.template : parser.error('Template not given')
    jsonData = opts.data
    jsonMig = opts.matrix
    jsonBkg = opts.background
    templateFile = opts.template
    modelName = opts.model
    verbose = opts.verbose

    pyfbu = pyFBU()
    pyfbu.nMCMC        = 100000
    pyfbu.nBurn        = 1000
    pyfbu.nThin        = 10
    pyfbu.lower        = 70000
    pyfbu.upper        = 140000
    pyfbu.jsonData     = jsonData
    pyfbu.jsonMig      = jsonMig
    pyfbu.jsonBkg      = jsonBkg
    pyfbu.templateFile = templateFile
    pyfbu.modelName    = modelName if modelName else pyfbu.modelName
    pyfbu.verbose      = verbose
    
    pyfbu.run()

    trace = pyfbu.trace

#    AcList  = computeAc.computeAcList(trace)
#    AcArray = np.array(AcList)
#    meanAc  = np.mean(AcArray)
#    stdAc   = np.std(AcArray)
