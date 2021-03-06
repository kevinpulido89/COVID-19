# COVID-19 -colombia
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

print(f'Today is {date.today().strftime("%d/%m/%Y")}.\n')

df = pd.read_excel('https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx')

TC = pd.DataFrame(columns=['Country', 'Total Cases', 'Total Deaths'])
pays = list(df['countriesAndTerritories'].unique())

for p in pays:
    tmp = df[df['countriesAndTerritories'] == p]['cases'].sum()
    tmp2 = df[df['countriesAndTerritories'] == p]['deaths'].sum()
    TC = TC.append({'Country': p,
                    'Total Cases': int(tmp),
                    'Total Deaths': int(tmp2)},
                   ignore_index=True)

TC.sort_values(by='Total Cases', ascending=False, axis=0, inplace=True)
TOP10 = TC.head(10)
TOP10.set_index('Country', inplace=True)
axesA = TOP10.plot.bar(rot=90, subplots=True)
plt.tight_layout()
# axesA[1].legend(loc=1)
plt.show()

LAT = pd.DataFrame()

paises_suramericanos = ['Mexico', 'Guatemala', 'Honduras', 'El_Salvador', 'Nicaragua', 'Panama',
                        'Colombia', 'Venezuela', 'Ecuador', 'Peru', 'Bolivia', 'Paraguay',
                        'Argentina', 'Uruguay', 'Cuba', 'Dominican_Republic', 'Guyana',
                        'Haiti', 'Costa_Rica', 'Chile', 'Brazil', 'Suriname']

for p in paises_suramericanos:
    tmp = df[df['countriesAndTerritories'] == p]['cases'].sum()
    tmp2 = df[df['countriesAndTerritories'] == p]['deaths'].sum()
    LAT = LAT.append({'Country': p,
                      'Total Cases': int(tmp),
                      'Total Deaths': int(tmp2)},
                     ignore_index=True)

LAT.sort_values(by='Total Cases', ascending=False, axis=0, inplace=True)
LAT.set_index('Country', inplace=True)
print(LAT)

axes = LAT.plot.bar(rot=90, subplots=True)
plt.tight_layout()
plt.show()
plt.close()

COL = df[df['countriesAndTerritories'] == 'Colombia']

# ###################---> FUNCIONES <---##########################

def acc_list(old):
    new = list()
    acc = 0
    for i in old:
        acc += i
        new.append(acc)
    return new


def cases_counter(_data_):
    cs = list(_data_['cases'])[::-1]
    return cs


def PlotPolly(model, independent_variable, dependent_variable, N):
    x_new = np.linspace(0, N, 100)
    y_new = model(x_new)

    New_case = model(N)

    tomorrow = model(N+1)

    # Prediccion para el siguente día
    print(f'Para el dia {len(dependent_variable)} se proyectan {int(p(len(dependent_variable)))} casos')
    print(f'Se estiman {int(tomorrow)} casos mañana')

    # plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-', label='Avance Casos COL')
    plt.plot(independent_variable, dependent_variable, '.', label='Casos confirmados')
    plt.plot(x_new, y_new, '-', label='Proyección Casos COL')
    plt.plot(N, int(New_case), color='r', marker='x', label='#Casos Esperados')
    plt.plot(N, 489122, color='g', marker='D', label='#Casos Reales')
    plt.title('Crecimiento de casos de COVID-19 en Colombia.')
    ax = plt.gca()
    ax.set_facecolor((0.892, 0.892, 0.892))
    fig = plt.gcf()
    plt.xlabel('Días')
    plt.ylabel('Cantidad de casos')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
##################---> FIN FUNCIONES <---########################


plt.plot(cases_counter(COL))
plt.xlabel("Días")
plt.ylabel("Casos")
plt.title('Aumento de casos diarios. Valor absoluto')
plt.tight_layout()
plt.show()
plt.close()

acc = acc_list(cases_counter(COL))
yy = [i for i in range(len(acc))]

acc_col = pd.DataFrame()
window_size = 3
for i in range(len(acc)-window_size):
    K = np.array(acc[i:i+window_size+1])
    acc_col = acc_col.append(pd.DataFrame(K).T)

# Here we use a polynomial of the 5th order
order = 6
f = np.polyfit(yy, acc, order)
# print('f: ',f)
p = np.poly1d(f)

print('\n')
print('El modelo ajustado de {0}° es:\n {1} \n'.format(order, p))

PlotPolly(p, yy, acc, len(acc))
