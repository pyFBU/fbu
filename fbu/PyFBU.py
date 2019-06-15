import pymc3 as mc
from numpy import random, dot, array, inf
import theano
import copy

class PyFBU(object):
    """A class to perform a MCMC sampling.

    [more detailed description should be added here]

    All configurable parameters are set to some default value, which
    can be changed later on, but before calling the `run` method.
    """
    #__________________________________________________________
    def __init__(self,data=[],response=[],background={},
                 backgroundsyst={},objsyst={'signal':{},'background':{}},
                 lower=[],upper=[],regularization=None,
                 rndseed=-1,verbose=False,name='',monitoring=False, mode=False):
        #                                     [MCMC parameters]
        self.nTune = 1000
        self.nMCMC = 10000 # N of sampling points
        self.nCores = 1 # number of CPU threads to utilize
        self.nChains = 2 # number of Markov chains to sample
        self.nuts_kwargs = None
        self.discard_tuned_samples = True # whether to discard tuning steps from posterior
        self.lower = lower  # lower sampling bounds
        self.upper = upper  # upper sampling bounds
        #                                     [unfolding model parameters]
        self.prior = 'Uniform'
        self.priorparams = {}
        self.obj_syst_flatprior = {'key': '__flat__', 'lower':-5, 'upper':5}
        self.freeze_NPs = {} # nuisance parameters for which to fix value (not sampled)
        self.regularization = regularization
        #                                     [input]
        self.data        = data           # data list
        self.response    = response       # response matrix
        self.background  = background     # background dict
        self.backgroundsyst = backgroundsyst
        self.objsyst        = objsyst
        self.systfixsigma = 0.
        #                                     [settings]
        self.rndseed   = rndseed
        self.verbose   = verbose
        self.name      = name
        self.monitoring = monitoring
        self.sampling_progressbar = True
        #                                     [mode]
        self.mode = mode
        self.MAP_method = 'L-BFGS-B'

    #__________________________________________________________
    def validateinput(self):
        def checklen(list1,list2):
            assert len(list1)==len(list2), 'Input Validation Error: inconstistent size of input'
        responsetruthbins = self.response
        responserecobins = [row for row in self.response]
        for bin in list(self.background.values())+responserecobins:
            checklen(self.data,bin)
        for bin in [self.lower,self.upper]:
            checklen(bin,responsetruthbins)
    #__________________________________________________________
    def fluctuate(self, data):
        random.seed(self.rndseed)
        return random.poisson(data)
    #__________________________________________________________
    def run(self):
        self.validateinput()
        data = self.data
        data = self.fluctuate(data) if self.rndseed>=0 else data

        # unpack background dictionaries
        backgroundkeys = self.backgroundsyst.keys()
        nbckg = len(backgroundkeys)

        backgrounds = []
        backgroundnormsysts = array([])
        if nbckg>0:
            backgrounds = array([self.background[key] for key in backgroundkeys])
            backgroundnormsysts = array([self.backgroundsyst[key] for key in backgroundkeys])

        # unpack object systematics dictionary
        objsystkeys = self.objsyst['signal'].keys()
        nobjsyst = len(objsystkeys)
        if nobjsyst>0:
            signalobjsysts = array([self.objsyst['signal'][key] for key in objsystkeys])
            if nbckg>0:
                backgroundobjsysts = array([])
                backgroundobjsysts = array([[self.objsyst['background'][syst][bckg]
                                             for syst in objsystkeys]
                                            for bckg in backgroundkeys])

        recodim  = len(data)
        resmat   = self.response
        truthdim = len(resmat)

        model = mc.Model()
        from .priors import wrapper
        add_kwargs = dict()
        if len(self.freeze_NPs) > 0:
            print('Freezing values of following NPs:')
            for key, val in self.freeze_NPs.items():
                print('{0}: {1}'.format(key, val))
        with model:
            truth = wrapper(priorname=self.prior,
                            low=self.lower,up=self.upper,
                            other_args=self.priorparams)

            if nbckg>0:
                bckgnuisances = []
                for name,err in zip(backgroundkeys,backgroundnormsysts):
                    try:
                        add_kwargs['observed'] = self.freeze_NPs[name]
                    except KeyError:
                        add_kwargs = dict()
                    if err<0.:
                        bckgnuisances.append(
                            mc.Uniform('norm_%s'%name,lower=0.,upper=3., **add_kwargs)
                            )
                    else:
                        # for fixed NP, one cannot use observed in bounded
                        # distribution, so we have to use unbounded one
                        if 'observed' in add_kwargs:
                            bckgnuisances.append(
                                mc.Normal('gaus_%s'%name, mu=0.,tau=1.0,
                                          **add_kwargs)
                            )
                        else:
                            BoundedNormal = mc.Bound(mc.Normal, lower=(-1.0/err if err>0.0 else -inf))
                            bckgnuisances.append(
                                BoundedNormal('gaus_%s'%name, mu=0.,tau=1.0)
                            )
                bckgnuisances = mc.math.stack(bckgnuisances)

            if nobjsyst>0:
                objnuisances = list()
                for name in objsystkeys:
                    try:
                        add_kwargs['observed'] = self.freeze_NPs[name]
                    except KeyError:
                        add_kwargs = dict()
                    if self.obj_syst_flatprior['key'] in name:
                        objnuisances.append(mc.Uniform('flat_%s'%name,
                                            lower=self.obj_syst_flatprior['lower'],
                                            upper=self.obj_syst_flatprior['upper'],
                                            **add_kwargs))
                    else:
                        objnuisances.append(mc.Normal('gaus_%s'%name,mu=0.,
                                                      tau=1.0, **add_kwargs))
                objnuisances = mc.math.stack(objnuisances)

        # define potential to constrain truth spectrum
            if self.regularization:
                truthpot = self.regularization.getpotential(truth)

        #This is where the FBU method is actually implemented
            def unfold():
                smearbckg = 1.
                if nbckg>0:
                    bckgnormerr = [(-1.+nuis)/nuis if berr<0. else berr
                                         for berr,nuis in zip(backgroundnormsysts,bckgnuisances)]
                    bckgnormerr = mc.math.stack(bckgnormerr)

                    smearedbackgrounds = backgrounds
                    if nobjsyst>0:
                        smearbckg = smearbckg + theano.dot(objnuisances,backgroundobjsysts)
                        smearedbackgrounds = backgrounds*smearbckg

                    bckg = theano.dot(1. + bckgnuisances*bckgnormerr,smearedbackgrounds)

                tresmat = array(resmat)
                reco = theano.dot(truth, tresmat)
                out = reco
                if nobjsyst>0:
                    smear = 1. + theano.dot(objnuisances,signalobjsysts)
                    out = reco*smear
                if nbckg>0:
                    out = bckg + out
                return out

            unfolded = mc.Poisson('unfolded', mu=unfold(),
                                  observed=array(data))

            import time
            from datetime import timedelta
            init_time = time.time()

            print(self.nuts_kwargs)


            if self.mode:
                map_estimate = mc.find_MAP(model=model, method=self.MAP_method)
                print (map_estimate)
                self.MAP = map_estimate
                self.trace = []
                self.nuisancestrace = []
                return

            trace = mc.sample(self.nMCMC,tune=self.nTune,cores=self.nCores,
                              chains=self.nChains, nuts_kwargs=self.nuts_kwargs,
                              discard_tuned_samples=self.discard_tuned_samples,
                              progressbar=self.sampling_progressbar)
            finish_time = time.time()
            print('Elapsed {0} ({1:.2f} samples/second)'.format(
                str(timedelta(seconds=(finish_time-init_time))).split('.')[0],
                (self.nMCMC+self.nTune)*self.nChains/(finish_time-init_time)
            ))

            self.trace = [trace['truth%d'%bin][:] for bin in range(truthdim)]
            #self.trace = [copy.deepcopy(trace['truth%d'%bin][:]) for bin in range(truthdim)]
            self.nuisancestrace = {}
            if nbckg>0:
                for name,err in zip(backgroundkeys,backgroundnormsysts):
                    try:
                        if err<0.:
                            self.nuisancestrace[name] = trace['norm_%s'%name][:]
                            #self.nuisancestrace[name] = copy.deepcopy(trace['norm_%s'%name][:])
                        if err>0.:
                            self.nuisancestrace[name] = trace['gaus_%s'%name][:]
                            #self.nuisancestrace[name] = copy.deepcopy(trace['gaus_%s'%name][:])
                    except KeyError as e:
                        try:
                            tmp = self.freeze_NPs[name]
                        except KeyError as e:
                            print('Warning: Missing NP trace', e)
            for name in objsystkeys:
                if self.systfixsigma==0.:
                    try:
                        if self.obj_syst_flatprior['key'] in name:
                            self.nuisancestrace[name] = trace['flat_%s'%name][:]
                        else:
                            self.nuisancestrace[name] = trace['gaus_%s'%name][:]
                        #self.nuisancestrace[name] = copy.deepcopy(trace['gaus_%s'%name][:])
                    except KeyError as e:
                        try:
                            tmp = self.freeze_NPs[name]
                        except KeyError as e:
                            print('Warning: Missing NP trace', e)

        if self.monitoring:
            from fbu import monitoring
            monitoring.plot(self.name+'_monitoring',data,backgrounds,resmat,self.trace,
                            self.nuisancestrace,self.lower,self.upper)
