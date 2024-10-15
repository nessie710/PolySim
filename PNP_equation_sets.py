import ufl


def assemble_stationary_problem(phi, c1, c2, vphi, v1, v2):
    F1 = -ufl.inner(ufl.grad(c1)[0], v1) * ufl.dx - ufl.inner(c1 * ufl.grad(phi)[0], v1) * ufl.dx
    F2 = -ufl.inner(ufl.grad(c1)[1], v1) * ufl.dx - ufl.inner(c1 * ufl.grad(phi)[1], v1) * ufl.dx
    F3 = -ufl.inner(ufl.grad(c1)[2], v1) * ufl.dx - ufl.inner(c1 * ufl.grad(phi)[2], v1) * ufl.dx
    F4 = ufl.inner(ufl.grad(c2)[0], v2) * ufl.dx - ufl.inner(c2 * ufl.grad(phi)[0], v2) * ufl.dx
    F5 = ufl.inner(ufl.grad(c2)[1], v2) * ufl.dx - ufl.inner(c2 * ufl.grad(phi)[1], v2) * ufl.dx
    F6 = ufl.inner(ufl.grad(c2)[2], v2) * ufl.dx - ufl.inner(c2 * ufl.grad(phi)[2], v2) * ufl.dx
    F7 = ufl.inner(ufl.grad(phi), ufl.grad(vphi)) * ufl.dx - (1/2)*ufl.inner((c1-c2), vphi) * ufl.dx
    F8 = ufl.inner(-ufl.grad(c1) - c1 * ufl.grad(phi), ufl.grad(v1)) * ufl.dx
    F9 = ufl.inner(ufl.grad(c2) - c2 * ufl.grad(phi), ufl.grad(v2)) * ufl.dx

    F = F1 + F2 + F3 + F4 + F5 + F6 + F7 + F8 + F9
    
    return F


def assemble_AC_problem(phiac, phi, c1ac, c2ac, c1, c2, vphiac, v1ac, v2ac, omega):
    # G1 = ufl.inner(c1ac*omega*1j, v1ac) * ufl.dx - ufl.inner(J1ac, ufl.grad(v1ac))*ufl.dx
    # G2 = ufl.inner(c2ac*omega*1j, v2ac) * ufl.dx + ufl.inner(J2ac, ufl.grad(v2ac))*ufl.dx
    G1 = ufl.inner(c1ac*omega*1j, v1ac) * ufl.dx - ufl.inner(-ufl.grad(c1ac) - c1ac*ufl.grad(phi) - c1*ufl.grad(phiac), ufl.grad(v1ac)) * ufl.dx
    G2 = ufl.inner(c2ac*omega*1j, v2ac) * ufl.dx + ufl.inner(ufl.grad(c2ac) - c2ac*ufl.grad(phi) - c2*ufl.grad(phiac), ufl.grad(v2ac)) * ufl.dx
    G3 = ufl.inner(ufl.grad(phiac), ufl.grad(vphiac))*ufl.dx - (1/2)*ufl.inner((c1ac - c2ac), vphiac)*ufl.dx 
    G7 = -ufl.inner(-ufl.grad(c1ac) - c1ac*ufl.grad(phi) - c1*ufl.grad(phiac) + ufl.grad(c2ac) - c2ac*ufl.grad(phi) - c2*ufl.grad(phiac)-1j*omega*ufl.grad(phiac), ufl.grad(vphiac)) * ufl.dx
    #G4 = ufl.inner(J1ac, vj1AC) * ufl.dx - ufl.inner(-ufl.grad(c1ac) - c1ac*ufl.grad(phi) - c1*ufl.grad(phiac), vj1AC) * ufl.dx
    #G5 = ufl.inner(J2ac, vj2AC) * ufl.dx - ufl.inner(ufl.grad(c2ac) - c2ac*ufl.grad(phi) - c2*ufl.grad(phiac), vj2AC) * ufl.dx

    G = G1 + G2 + G3 + G7
    
    return G