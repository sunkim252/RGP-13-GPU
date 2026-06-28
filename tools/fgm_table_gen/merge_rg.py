#!/usr/bin/env python3
# Tier-2 for the 4-D enthalpy table: the 13 RG_* mixing coefficients are
# composition-only (calculateRealGas takes X, no T/p) and the 4-D table freezes
# composition across dh -> replicate each 3-D RG node value nH times (dh is the
# innermost flat index) and append to the 4-D dict so the solver tabulates RG
# instead of doing the live O(n^2) Chung/SRK mixing.
import re, sys, numpy as np

SRC3D = sys.argv[1]   # constant/fgmProperties.3d_bak  (has RG_*)
DST4D = sys.argv[2]   # constant/fgmProperties         (4-D, gets RG_*)
NH    = int(sys.argv[3]) if len(sys.argv) > 3 else 11

names = ['bM','cM','coef1','coef2','coef3','epsilonkM','kappaiM',
         'miuiM','MM','omegaM','sigmaM','TcM','VcM']

txt3 = open(SRC3D).read()
# sanity: 4-D dict must not already have RG_*
d4 = open(DST4D).read()
if 'RG_bM' in d4:
    print('4-D dict already has RG_* -- abort'); sys.exit(1)

blocks = []
n3 = None
for nm in names:
    m = re.search(r'RG_%s\b[^(]*\(([^)]*)\)' % re.escape(nm), txt3)
    if not m:
        print('MISSING RG_%s in 3-D dict' % nm); sys.exit(1)
    vals = np.fromstring(m.group(1), sep=' ')
    if n3 is None: n3 = vals.size
    if vals.size != n3:
        print('size mismatch RG_%s: %d vs %d' % (nm, vals.size, n3)); sys.exit(1)
    rep = np.repeat(vals, NH)             # dh innermost -> each value x NH
    body = ' '.join('%.9g' % v for v in rep)
    blocks.append('RG_%s %d\n(\n%s\n);\n' % (nm, rep.size, body))
    print('RG_%s: %d -> %d' % (nm, vals.size, rep.size))

ins = '\n'.join(blocks)
idx = d4.rfind('// *****')
if idx < 0:
    idx = len(d4)
open(DST4D, 'w').write(d4[:idx] + ins + '\n' + d4[idx:])
print('done: appended 13 RG_* (3-D n=%d -> 4-D n=%d) to %s' % (n3, n3*NH, DST4D))
