from tcplDelete import tcplDelete
from tcplLoadAeid import tcplLoadAeid

def tcplCascade(lvl, id):
    if lvl == 0:
        tcplDelete(tbl="mc0", fld="acid", val=id)
    if lvl <= 1:
        tcplDelete(tbl="mc1", fld="acid", val=id)
    if lvl <= 2:
        tcplDelete(tbl="mc2", fld="acid", val=id)
    if lvl < 3:
        try:
            id = tcplLoadAeid("acid", id)["aeid"]
        except:
            return True
    if lvl <= 3:
        tcplDelete(tbl="mc3", fld="aeid", val=id)
    if lvl <= 4:
        tcplDelete(tbl="mc4_", fld="aeid", val=id)
    if lvl <= 4:
        tcplDelete(tbl="mc4_agg_", fld="aeid", val=id)
    if lvl <= 4:
        tcplDelete(tbl="mc4_param_", fld="aeid", val=id)
    if lvl <= 5:
        tcplDelete(tbl="mc5_", fld="aeid", val=id)
    if lvl <= 5:
        tcplDelete(tbl="mc5_param_", fld="aeid", val=id)
    if lvl <= 6:
        tcplDelete(tbl="mc6_", fld="aeid", val=id)

    print("Completed delete cascade for", len(id), "ids\n")
