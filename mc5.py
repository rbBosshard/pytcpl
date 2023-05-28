from tcplLoadData import load_mc4_data
from tcplWriteData import tcplWriteData
from tcplMthdLoad import tcplMthdLoad
from query_db import tcplQuery
from mc5_mthds import mc5_mthds
from tcplHit2 import tcplHit2

ae = 5
# Load level 4 data
df = load_mc4_data(ae).head(100)
print(f"df_shape: {df.shape}")

# Check if any level 4 data was loaded
if df.shape[0] == 0:
    warning_msg = f"No level 4 data for AEID {ae}. Level 5 processing incomplete; no updates made to the mc5 table for AEID {ae}."
    print(warning_msg)

print(f"Loaded L4 AEID {ae} ({df.shape[0]} rows)")

ms = tcplMthdLoad(lvl = 5, id = ae, type = "mc")
mthd_funcs = mc5_mthds(ae)

coff = []
for method_key in ms['mthd'].tolist():
    coff.append(mthd_funcs.get(method_key)(df))

print(f"coff: {coff}")
if ms.shape[0] == 0:
    warning_msg = f"No level 5 methods for AEID {ae} -- cutoff will be 0."
    print(warning_msg)


# Determine final cutoff
max_coff = max(coff)
print(f'max_coff: {max_coff}')
df['coff'] = max_coff[0]

cutoff = max(df['coff'])
# can remove this once loading of data is working correctly
mc4_name = "mc4_"
mc5_name = "mc5_"
mc4_param_name = "mc4_param_"

query = f"SELECT {mc4_name}.m4id, {mc4_name}.aeid, {mc4_name}.spid, {mc4_name}.bmad, {mc4_name}.resp_max, {mc4_name}.resp_min, {mc4_name}.max_mean, {mc4_name}.max_mean_conc, {mc4_name}.max_med, {mc4_name}.max_med_conc, {mc4_name}.logc_max, {mc4_name}.logc_min, {mc4_name}.nconc, {mc4_name}.npts, {mc4_name}.nrep, {mc4_name}.nmed_gtbl, {mc4_name}.tmpi, {mc4_param_name}.model, {mc4_param_name}.model_param, {mc4_param_name}.model_val FROM {mc4_name} INNER JOIN {mc4_param_name} ON {mc4_name}.m4id = {mc4_param_name}.m4id WHERE {mc4_name}.aeid = {ae};"

dat = tcplQuery(query)
print(f"dat: {dat.shape}")

# if we're using v3 schema we want to tcplfit2
dat = tcplHit2(dat, coff=cutoff)

print(f"dat---: {dat.shape}")
# tcplWriteData(dat = dat, lvl = 5, type = "mc")

print(f"Processed L5 AEID {ae} ({dat.shape[0]} rows)")



   