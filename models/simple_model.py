def rop(wob, wob_opt):
    return 1/wob_opt*(wob_opt**2-((wob - wob_opt))**2)

def rop_multi(wob,rpm,q,wob_opt,rpm_opt, q_opt):
    return ((1/wob_opt)*(wob_opt**2-((wob - wob_opt))**2)) + ((1/rpm_opt)*(rpm_opt**2-((rpm - rpm_opt))**2)) + ((1/q_opt)*(q_opt**2-((q - q_opt))**2))

def rop_statements(wob,rpm,q,wob_opt,rpm_opt, q_opt):
    if wob >= 0 and rpm >= 0 and q >= 0:
        return ((1/wob_opt)*(wob_opt**2-((wob - wob_opt))**2)) * ((1/rpm_opt)*(rpm_opt**2-((rpm - rpm_opt))**2)) * ((1/q_opt)*(q_opt**2-((q - q_opt))**2))
    else:
        return -2*abs(((1/wob_opt)*(wob_opt**2-((wob - wob_opt))**2)) * ((1/rpm_opt)*(rpm_opt**2-((rpm - rpm_opt))**2)) * ((1/q_opt)*(q_opt**2-((q - q_opt))**2)))