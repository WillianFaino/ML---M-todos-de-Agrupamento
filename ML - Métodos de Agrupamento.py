import math as ma
import copy
import itertools
import collections
import numpy as np
import pandas as pd
import matplotlib as m
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering



dados = pd.read_csv('H:\Downloads\Base4.csv')

def naoseimexercomdataframe(dados):
    ldadosX = []
    ldadosY = []
    labelDados=[]

    for i in dados.iterrows():
        ldadosX.append(float(i[1][0]))
        ldadosY.append(float(i[1][1]))
        labelDados.append(i[1][2])

    return [ldadosX, ldadosY, labelDados]

def mean(v, labels, dbscan=False):
    s = 0
    lbn = len(np.unique(labels))

    if dbscan:
        lbn-=1

    for i in range(len(v)):
        s+=v[i]


    # print(s)
    # print(lbN)

    return (s/lbn)

def cohesion(lx, ly, lb):
    lilistasX = []
    lilistasY = []
    cohelist = []

    for i in range(20):
        lilistasX.append([])
        lilistasY.append([])
        cohelist.append(0)

    for i in range(2000):
        if lb[i]<20:
            if lb[i]!=-1:
                lilistasX[int(lb[i])].append(lx[i])
                lilistasY[int(lb[i])].append(ly[i])

    for i in range(20):
        if len(lilistasX[i])>0:
            # print("lilistaX[i]", lilistasX[i])
            for j in range(len(lilistasX[i])):
                for k in range(len(lilistasY[i])):
                    a=ma.sqrt((lilistasX[i][j]-lilistasX[i][k])**2)+((lilistasY[i][j]-lilistasY[i][k])**2)
                    b=(len(lilistasX[i]))**2
                    cohelist[i] += a/b
                    # print("a/b = ", a/b)


    return cohelist

def separation(lx, ly, lb):
    lilistasX = []
    lilistasY = []
    sepalist = []
    clun = 0

    for i in range(20):
        lilistasX.append([])
        lilistasY.append([])
        sepalist.append(0)

    for i in range(2000):
        if lb[i]<20:
            if lb[i]!=-1:
                lilistasX[int(lb[i])].append(lx[i])
                lilistasY[int(lb[i])].append(ly[i])

    for i in range(20):
        if len(lilistasX[i])>0:
            clun+=1

    for i in range(20):
        if len(lilistasX[i])>0:
            for l in range(20):
                if len(lilistasX[l]) > 0:
                    if l!=i:
                        # print("l = ", l)
                        for j in range(len(lilistasX[l])):
                            for k in range(len(lilistasY[i])):
                                a=ma.sqrt(((lilistasX[l][j]-lilistasY[i][k])**2)+((lilistasY[l][j]-lilistasX[i][k])**2))
                                # print("a = ", a)
                                b=((len(lilistasX[i]))*(len(lilistasX[l])))
                                # print("b = ", b)]
                                # if len(lilistasX[i])==1:
                                     # print("sepalist = ", sepalist)
                                sepalist[i] += a/b
        sepalist[i]/=clun-1
    # print("sepalist = ", sepalist)
    return sepalist

def entropy(origLabels, clusteringLabels):
    lilistasL = []
    entrolist = []

    for i in range(200):
        lilistasL.append([])
        entrolist.append([])

    for i in range(len(origLabels)):
        lilistasL[int(clusteringLabels[i])].append(origLabels[i])



    for i in range(len(lilistasL)):
        if len(lilistasL[i])>0:
            lbn = collections.Counter(lilistasL[i])
            for j in range(len(origLabels)):
                entrolist[i].append(lbn[j]/len(lilistasL[i]))


    entrolistAC = []

    for i in range(len(lilistasL)):
        if len(entrolist[i])>0:
            ent=0
            for j in range(len(origLabels)):
                if entrolist[i][j]!=0:
                    ent -= entrolist[i][j] * ma.log(entrolist[i][j], 2)
            entrolistAC.append(ent)

    return sum(entrolistAC)/(len(np.unique(clusteringLabels))-1)









lista = naoseimexercomdataframe(dados = dados)
dadosLX = lista[0]
dadosLY = lista[1]
labelsL = lista[2]




attSet = dados.drop("label", axis=1)
labelSet = dados["label"]

print(np.unique(labelSet))  #show all different labels

print(dados.groupby('label').size()) #show the quantitiy of instances of each label




#some database avaliation

# database_cohesionvec = cohesion(dadosLX,dadosLY,labelsL)
#
# print(database_cohesionvec)
#
# database_cohesion = mean(database_cohesionvec, labelsL)
#
# print("database cohesion = ", database_cohesion)
#
# database_separationvec = separation(dadosLX, dadosLY, labelsL)
#
# print(database_separationvec)
#
# database_separation = mean(database_separationvec, labels = labelsL)
#
# print("database separation = ", database_separation)











##################################        DBSCAN      ########################################
# dbs = DBSCAN(eps=0.9, min_samples=20)
# dbs.fit(attSet)
# f,(dbsg, origg) = plt.subplots(1,2,sharey=True,figsize=(20,8))
# dbsg.set_title("DBScan")
# dbsg.scatter(dadosLX, dadosLY,c=dbs.labels_, cmap="rainbow")
# origg.set_title("Original")
# origg.scatter(dadosLX, dadosLY,c=labelsL, cmap="rainbow")
# dbs_cohesionvec = cohesion(dadosLX, dadosLY, dbs.labels_)
# # print(dbs_cohesionvec)
# dbs_cohesion = mean(dbs_cohesionvec, dbscan=True, labels=dbs.labels_)
# print("DBSCAN cohesion = ", dbs_cohesion)
# dbs_separationvec = separation(dadosLX, dadosLY, dbs.labels_)
# dbs_separation = mean(dbs_separationvec, dbscan=True, labels=dbs.labels_)
# print("DBSCAN separation = ", dbs_separation)
# dbs_entropy = entropy(labelsL, dbs.labels_)
# print("DBSCAN entropy = ", dbs_entropy)
# dbs_SiScore = silhouette_score(dados, dbs.labels_)
# print("Coeficiente de Silhueta DBSCAN = ", dbs_SiScore)
# plt.show([dbsg,origg])









##################################        K-Means      #######################################
# kmns = KMeans(n_clusters=9, max_iter=27)
# kmns.fit(attSet)
# f,(kmnsg, origg) = plt.subplots(1,2,sharey=True,figsize=(20,8))
# kmnsg.set_title("K-Means")
# kmnsg.scatter(dadosLX,dadosLY,c=kmns.labels_,cmap="rainbow")
# origg.set_title("Original")
# origg.scatter(dadosLX,dadosLY,c=labelsL,cmap="rainbow")
# kmns_cohesionvec = cohesion(dadosLX, dadosLY, kmns.labels_)
## print(kmns_cohesionvec)
# kmns_cohesion = mean(kmns_cohesionvec, labels=kmns.labels_)
# print("K-Means cohesion = ", kmns_cohesion)
## kmns_separationvec = separation(dadosLX, dadosLY, kmns.labels_)
# kmns_separation = mean(kmns_separationvec, labels=kmns.labels_)
# print("K-Means separation = ", kmns_separation)
# kmns_entropy = entropy(labelsL, kmns.labels_)
# print("K-Means entropy = ", kmns_entropy)
# kmns_SiScore = silhouette_score(dados, kmns.labels_)
# print("Coeficiente de Silhueta K-Means = ", kmns_SiScore)
# plt.show([kmnsg,origg])













##################################        AGNES      #########################################
ag = AgglomerativeClustering(n_clusters=3, linkage="single")
ag.fit(attSet)
f,(agg, origg) = plt.subplots(1,2,sharey=True,figsize=(20,8))
agg.set_title("AGNES - Agglomerative Clustering")
agg.scatter(dadosLX,dadosLY,c=ag.labels_,cmap="rainbow")
origg.set_title("Original")
origg.scatter(dadosLX,dadosLY,c=labelsL,cmap="rainbow")
ag_cohesionvec = cohesion(dadosLX, dadosLY, ag.labels_)
## print(ag_cohesionvec)
ag_cohesion = mean(ag_cohesionvec, labels=ag.labels_)
print("AGNES cohesion = ", ag_cohesion)
ag_separationvec = separation(dadosLX, dadosLY, ag.labels_)
## print(ag_separationvec)
ag_separation = mean(ag_separationvec, labels=ag.labels_)
print("AGNES separation = ", ag_separation)
ag_entropy = entropy(labelsL, ag.labels_)
print("AG entropy = ", ag_entropy)
ag_SiScore = silhouette_score(dados, ag.labels_)
print("Coeficiente de Silhueta AG = ", ag_SiScore)
plt.show([agg,origg])
