from rate_of_penetration import rate_of_penetration
from generate_parameter import generate_range


def simulation(depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, depth_final, delta_t):
    ##initialize parameters
    a1,a2,a3,a4,a5,a6,a7,a8 = generate_range()
    t = 0.0

    ## initialize return dictionaries
    time_dict = []
    depth_dict = []
    parameter_dict = []
    rop_dict = []

    ##start simulation
    while depth < depth_final:
        rop = rate_of_penetration(a1,a2,a3,a4,a5,a6,a7,a8,depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v)
        depth += rop*delta_t
        t += delta_t
        depth_dict.append(depth)
        rop_dict.append(rop)
        time_dict.append(t)

    return time_dict, depth_dict, rop_dict
    
