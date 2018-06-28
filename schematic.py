import numpy as np
import argparse

class correlator (object):

    def __init__ (self, blocksize=256, h=1e-9, blocks=60,
                  maxiter=10000, accuracy=1e-9,
                  nu=1.0, kernel = lambda phi: 0.0, base = None):
        if base is None:
            self.h0 = 1e-9
            self.blocks = blocks
            self.halfblocksize = int(blocksize/4)*2
            self.blocksize = self.halfblocksize * 2
        else:
            self.h0 = base.h0
            self.blocks = base.blocks
            self.halfblocksize = base.halfblocksize
            self.blocksize = base.blocksize
            self.base = base
        self.h = self.h0
        self.phi = np.zeros(self.blocksize)
        self.m = np.zeros(self.blocksize)
        self.dPhi = np.zeros(self.halfblocksize+1)
        self.dM = np.zeros(self.halfblocksize+1)

        self.maxiter = maxiter
        self.accuracy = accuracy

        self.nu = nu
        self.kernel = kernel


    def initial_values (self, imax=50):
        iend = imax
        if (iend > self.halfblocksize): iend = self.halfblocksize
        for i in range(iend):
             t = i*self.h0
             self.phi[i] = 1.0 - t/self.nu
             self.m[i] = self.kernel (self.phi[i], i, t)
        for i in range(1,iend):
             self.dPhi[i] = 0.5 * (self.phi[i-1] + self.phi[i])
             self.dM[i] = 0.5 * (self.m[i-1] + self.m[i])
        self.dPhi[iend] = self.phi[iend-1]
        self.dM[iend] = self.m[iend-1]
        self.iend = iend

    def decimize (self):
        imid = int(self.halfblocksize/2)
        for i in range(1,imid):
            di = i+i
            self.dPhi[i] = 0.5 * (self.dPhi[di-1] + self.dPhi[di])
            self.dM[i] = 0.5 * (self.dM[di-1] + self.dM[di])
        for i in range(imid,self.halfblocksize):
            di = i+i
            self.dPhi[i] = 0.25 * (self.phi[di-2] + 2*self.phi[di-1] + self.phi[di])
            self.dM[i] = 0.25 * (self.m[di-2] + 2*self.m[di-1] + self.m[di])
        self.dPhi[self.halfblocksize] = self.phi[self.blocksize-1]
        self.dM[self.halfblocksize] = self.m[self.blocksize-1]
        for i in range(self.halfblocksize):
            di = i+i
            self.phi[i] = self.phi[di]
            self.m[i] = self.m[di]
        self.h = self.h * 2.0


    def solve_block (self, istart, iend):
        nutmp = self.nu/self.h
        A = 1.0 + self.dM[1] + 1.5*nutmp
        B = (1.0 - self.dPhi[1]) / A

        for i in range(istart,iend):
            ibar = int(i/2)
            C = -(self.m[i-1]*self.dPhi[1] + self.phi[i-1]*self.dM[1])
            for k in range(2,ibar+1):
                C += (self.m[i-k+1] - self.m[i-k]) * self.dPhi[k]
                C += (self.phi[i-k+1] - self.phi[i-k]) * self.dM[k]
            if (i-ibar > ibar):
                C += (self.phi[i-ibar] - self.phi[i-ibar-1]) * self.dM[k]
            C += self.m[i-ibar] * self.phi[ibar]
            C += (-2.0*self.phi[i-1] + 0.5*self.phi[i-2]) * nutmp
            C = C/A

            iterations = 0
            converged = False
            newphi = self.phi[i-1]
            while (not converged and iterations < self.maxiter):
                self.phi[i] = newphi
                self.m[i] = self.kernel (self.phi[i], i, self.h*i)
                newphi = B*self.m[i] - C
                iterations += 1
                if np.isclose (newphi, self.phi[i],
                               rtol=self.accuracy, atol=self.accuracy):
                    converged = True
                    self.phi[i] = newphi


class mean_squared_displacement (correlator):

    def initial_values (self, imax=50):
        iend = imax
        if (iend > self.halfblocksize): iend = self.halfblocksize
        for i in range(iend):
            t = i*self.h0
            self.phi[i] = 6*t/self.nu
            self.m[i] = self.kernel (None, i, t)
        for i in range(1,iend):
            self.dPhi[i] = 0.5 * (self.phi[i-1] + self.phi[i])
            self.dM[i] = 0.5 * (self.m[i-1] + self.m[i])
        self.dPhi[iend] = self.phi[iend-1]
        self.dM[iend] = self.m[iend-1]
        self.iend = iend

    def solve_block (self, istart, iend):
        nutmp = self.nu/self.h
        A = self.dM[1] + 1.5*nutmp

        for i in range(istart,iend):
            ibar = int(i/2)
            C = (self.m[i]-self.m[i-1])*self.dPhi[1] - self.phi[i-1]*self.dM[1]
            for k in range(2,ibar+1):
                C += (self.m[i-k+1] - self.m[i-k]) * self.dPhi[k]
                C += (self.phi[i-k+1] - self.phi[i-k]) * self.dM[k]
            if (i-ibar > ibar):
                C += (self.phi[i-k+1] - self.phi[i-k]) * self.dM[k]
            C += self.m[i-ibar] * self.phi[ibar]
            C += (-2.0*self.phi[i-1] + 0.5*self.phi[i-2]) * nutmp
            C += -6.0
            C = C/A

            self.m[i] = self.kernel (None, i, self.h*i)
            self.phi[i] = - C


def output (istart, iend, correlator_array, filter=0):
    first = correlator_array[0]
    for i in range(istart,iend):
        if not (filter and (i%filter)):
            print ("{t:.15f} ".format(t=i*first.h),end='')
            for correlator in correlator_array:
                print ("{phi:.15f} ".format(phi=correlator.phi[i]),end='')
            print ("#")




class nonergodicity_parameter (object):

    def __init__ (self, model, accuracy=1e-9, maxiter=1000000):
        self.model = model
        self.accuracy = accuracy
        self.maxiter = maxiter

    def solve (self):
        kernel = self.model.m
        iterations = 0
        converged = False
        newf = 1.0
        while (not converged and iterations < self.maxiter):
          self.f = newf
          self.m = kernel (self.f)
          newf = self.m / (1.0+self.m)
          if np.isclose (newf, self.f, rtol=self.accuracy, atol=0.0):
            converged = True
            self.f = newf


class eigenvalue (object):

    def __init__ (self, nep, accuracy=1e-9, maxiter=1000000):
        self.nep = nep
        self.accuracy = accuracy
        self.maxiter = maxiter

    def solve (self):
        f = self.nep.f
        model = self.nep.model
        iterations = 0
        converged = False
        newe = 1.0
        while (not converged and iterations < self.maxiter):
            self.e = newe
            newe = (1-f)*model.dm(f,self.e)*(1-f)
            norm = np.abs(newe)
            if norm>1e-10: newe = newe/norm
            if np.isclose (newe, self.e, rtol=self.accuracy, atol=0.0):
                converged = True
                self.e = newe
                self.eval = norm
        self.ehat = self.e
        if self.eval > 0:
            nl = self.ehat * self.e
            nr = self.ehat * self.e*self.e / (1-f)
            self.e = self.e * nl/nr
            self.ehat = self.ehat * nr/(nl*nl)
            self.lam = self.ehat * (1-f)*model.dm2 (f,self.e)*(1-f)
            #nr = self.ehat * self.e*self.e / (1-f)
            #print ("nr {}".format(nr))
            #self.lam = self.lam / nr
        else:
            self.lam = 0.0


class f12model (object):
    def __init__ (self, v1, v2):
        self.v1 = v1
        self.v2 = v2
    def m (self, phi, i=0, t=0):
        return self.v1 * phi + self.v2 * phi*phi
    def dm (self, phi, dphi):
        return self.v1 * dphi + 2 * self.v2 * phi*dphi
    def dm2 (self, phi, dphi):
        return self.v2 * dphi*dphi

class sjoegren_model (object):
    def __init__ (self, vs, base_correlator):
        self.vs = vs
        self.base = base_correlator
    def m (self, phi, i=0, t=0):
        return self.vs * phi * self.base.phi[i]

class msd_model (object):
    def __init__ (self, vs, base_correlator):
        self.vs = vs
        self.base = base_correlator
    def m (self, phi, i=0, t=0):
        return self.vs * self.base.phi[i] * self.base.base.phi[i]



class f12gammadot_model (object):
    def __init__ (self, v1, v2, gammadot=0.0, gammac=0.1):
        self.v1 = v1
        self.v2 = v2
        self.gammadot = gammadot
        self.gammac = gammac
    def m (self, phi, i, t):
        gt = self.gammadot * t / self.gammac
        return (self.v1 * phi + self.v2 * phi*phi) * 1.0/(1.0 + gt*gt)



parser = argparse.ArgumentParser()
parser.add_argument ('-v1',metavar='v1',help='vertex v1 of F12 model',
                     type=float, default=0.8)
parser.add_argument ('-v2',metavar='v2',help='vertex v2 of F12 model',
                     type=float, default=2.0)
parser.add_argument ('-vs',metavar='vs',help='vertex vs of Sjoegren model',
                     type=float, default=15.0)
parser.add_argument ('-gammadot',metavar='gdot',help='shear rate',
                     type=float, default=1e-4)
args = parser.parse_args()



model = f12model(args.v1,args.v2)
phi = correlator (kernel = model.m)

model_s = sjoegren_model(args.vs,phi)
phi_s = correlator (kernel = model_s.m, base=phi)

model_msd = msd_model(args.vs,phi_s)
msd = mean_squared_displacement (kernel = model_msd.m, base=phi)

shear_model = f12gammadot_model(model.v1,model.v2,gammadot=args.gammadot)
phi_gdot = correlator (kernel = shear_model.m)

f = nonergodicity_parameter (model)
f.solve()
print ("# f = {:f}".format(f.f))

eval = eigenvalue (f)
eval.solve()
print ("# eigenvalue = {:f}".format(eval.eval))
print ("# e = {:f}".format(eval.e))
print ("# ehat = {:f}".format(eval.ehat))
print ("# lambda = {:f}".format(eval.lam))

#quit()

blocksize = phi.blocksize
halfblocksize = phi.halfblocksize
blocks = phi.blocks

correlators = [phi, phi_s, msd, phi_gdot]

for _phi_ in correlators:
    _phi_.initial_values ()
    _phi_.solve_block (_phi_.iend, halfblocksize)
output (0, halfblocksize, correlators)

for d in range(blocks):
    for _phi_ in correlators:
        _phi_.solve_block (_phi_.halfblocksize, _phi_.blocksize)
    output (halfblocksize, blocksize, correlators)
    for _phi_ in correlators:
        _phi_.decimize ()
