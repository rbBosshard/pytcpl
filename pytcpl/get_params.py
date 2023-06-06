

def get_params(fitmethod):
    params = dict(cnst=['er'], poly1=['a', 'er'], poly2=['a', 'b', 'er'], pow=['a', 'p', 'er'], exp2=['a', 'b', 'er'],
                  exp3=['a', 'b', 'p', 'er'], exp4=['tp', 'ga', 'er'], exp5=['tp', 'ga', 'p', 'er'],
                  hill=['tp', 'ga', 'p', 'er'], hill_=['tp', 'ga', 'p', 'er'], gnls=['tp', 'ga', 'p', 'la', 'q', 'er'],
                  gnls_=['tp', 'ga', 'p', 'la', 'q', 'er']).get(fitmethod)
    return params
