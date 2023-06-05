from tcplDelete import tcplDelete

def tcplCascade(lvl, id):
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

    print("Completed delete cascade for", len(id), "ids\n")
