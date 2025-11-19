def age(stream):
    if stream == 'AAU':
        return 12
    else:
        return 'None'
def z(stream):
    if stream == 'AAU':
        return 0.0007
    else:
        return 'None'
def get_mu(stream):
    if stream == 'AAU':
        def mu(phi1):
            return 16.727 - 0.0282 * phi1 + 0.00018 * (phi1 ** 2)
    else:
        return 'Nope'
