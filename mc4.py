from tcplFit2 import tcplFit2
from tcplLoadData import load_mc3_data
from tcplMthdLoad import tcplMthdLoad
from mc4_mthds import mc4_mthds
from tcplWriteData import tcplWriteData

id = 5
# fitmodels = ["cnst", "hill", "gnls", "poly1", "poly2", "pow", "exp2", "exp3", "exp4", "exp5"]
fitmodels = ["cnst"]

df = load_mc3_data(id) #.head(100)
ms = tcplMthdLoad(lvl = 4, id = id, type = "mc")

if (ms.shape[0] == 0):
    print(f"No level 4 methods for AEID {id} Level 4 processing incomplete; no updates made to the mc4.")

mthd_funcs = mc4_mthds()

for method_key in ms['mthd'].tolist():
    df = mthd_funcs.get(method_key)(df)

# fit
bidirectional = True
df = tcplFit2(df, fitmodels=fitmodels, bidirectional=bidirectional)

tcplWriteData(dat = df, lvl = 4)

print(df.head(1))