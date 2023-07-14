from multiprocessing import Pool
import pandas as pd
import numpy as np
import json
from collections import OrderedDict,Counter
from Scores import *
from sklearn import preprocessing 

def Dataprocessing(args,fasta_dict,testfasta_dict,test_labels,predictfasta_dict,allmatrixName,L):
    global length
    length=L
    coding_scheme = ['WLS', 'IC50', 'AAF', 'PWM', 'PPM', 'NNS', 'KNN', 'BSI', 'PAM']
    
    if args.processNum == None:
        data, allmatrix = get_features(args.protease, fasta_dict, coding_scheme, False, None,length)
    else: # 
        Count = Counter(fasta_dict.values())
        labels = [1] * int(Count['1']) + [0] * int(Count['0'])
        labelDF = pd.DataFrame()
        labelDF['true_label'] = labels

        p = Pool(args.processNum)

        result = p.starmap(get_features1, [(args.protease, fasta_dict, 'WLS', False, None,length),
                                            (args.protease, fasta_dict, 'IC50', False, None,length),
                                            (args.protease, fasta_dict, 'AAF', False, None,length),
                                            (args.protease, fasta_dict, 'PWM', False, None,length),
                                            (args.protease, fasta_dict, 'PPM', False, None,length),
                                            (args.protease, fasta_dict, 'NNS', False, None,length),
                                            (args.protease, fasta_dict, 'KNN', False, None,length),
                                            (args.protease, fasta_dict, 'BSI', False, None,length),
                                            (args.protease, fasta_dict, 'PAM', False, None,length),
                                            ])
        p.close()
        p.join()
        data = pd.concat(
            [labelDF, result[0][0], result[1][0], result[2][0], result[3][0], result[4][0], result[5][0],
                result[6][0], result[7][0], result[8][0]], axis=1)

        allmatrix = OrderedDict()
        allmatrix[args.protease] = OrderedDict()
        for i, coding in enumerate(coding_scheme):
            if coding == 'BSI':
                for bls in blosumDict.keys():
                    allmatrix[args.protease][bls] = result[i][1][args.protease][bls]
            elif coding == 'PAM':
                for pam in PAMDict.keys():
                    allmatrix[args.protease][pam] = result[i][1][args.protease][pam]
            else:
                allmatrix[args.protease][coding] = result[i][1][args.protease][coding]
    # # save matrix
    with open(allmatrixName, 'w') as f:
        json.dump(allmatrix, f)
        
    if args.testfile:  
        if args.processNum == None:
            labelDF1 = pd.DataFrame()
            labelDF1['true_label'] = test_labels
            testdata= get_features(args.protease, testfasta_dict, coding_scheme, True, allmatrix,length)
            testdata = pd.concat(
                [labelDF1, testdata],
                axis=1)
        else:

            labelDF1 = pd.DataFrame()
            labelDF1['true_label'] = test_labels
            p1 = Pool(args.processNum)
            
            result1 = p1.starmap(get_features1, [(args.protease, testfasta_dict, 'WLS', True, allmatrix,length),
                                                    (args.protease, testfasta_dict, 'IC50', True, allmatrix,length),
                                                    (args.protease, testfasta_dict, 'AAF', True, allmatrix,length),
                                                    (args.protease, testfasta_dict, 'PWM', True, allmatrix,length),
                                                    (args.protease, testfasta_dict, 'PPM', True, allmatrix,length),
                                                    (args.protease, testfasta_dict, 'NNS', True, allmatrix,length),
                                                    (args.protease, testfasta_dict, 'KNN', True, allmatrix,length),
                                                    (args.protease, testfasta_dict, 'BSI', True, allmatrix,length),
                                                    (args.protease, testfasta_dict, 'PAM', True, allmatrix,length)
                                                    ])
            p1.close()
            p1.join()

            testdata = pd.concat(
                [labelDF1, result1[0][0], result1[1][0], result1[2][0], result1[3][0], result1[4][0],
                    result1[5][0], result1[6][0], result1[7][0], result1[8][0]],
                axis=1)
                    
    if predictfasta_dict:
        if args.processNum == None:
            predata= get_features(args.protease, predictfasta_dict, coding_scheme, True, allmatrix,length)
        else:

            p2 = Pool(args.processNum)
            result2 = p2.starmap(get_features1, [(args.protease, predictfasta_dict, 'WLS', True, allmatrix,length),
                                                    (args.protease, predictfasta_dict, 'IC50', True, allmatrix,length),
                                                    (args.protease, predictfasta_dict, 'AAF', True, allmatrix,length),
                                                    (args.protease, predictfasta_dict, 'PWM', True, allmatrix,length),
                                                    (args.protease, predictfasta_dict, 'PPM', True, allmatrix,length),
                                                    (args.protease, predictfasta_dict, 'NNS', True, allmatrix,length),
                                                    (args.protease, predictfasta_dict, 'KNN', True, allmatrix,length),
                                                    (args.protease, predictfasta_dict, 'BSI', True, allmatrix,length),
                                                    (args.protease, predictfasta_dict, 'PAM', True, allmatrix,length),
                                                    ])
            p2.close()
            p2.join()

            predata = pd.concat(
                [result2[0][0], result2[1][0], result2[2][0], result2[3][0], result2[4][0],
                    result2[5][0], result2[6][0], result2[7][0], result2[8][0]],
                axis=1) 

        if args.testfile:
            return data,testdata,predata
        else:
            return data,data,predata
    else:
        if args.testfile:
            return data,testdata
        else:
            return data,data   

def Pre_processing(args,fasta_dict,allmatrix,proteasesOne,L):
    global length
    length=L

    coding_scheme = ['WLS', 'IC50', 'AAF', 'PWM', 'PPM', 'NNS', 'KNN', 'BSI', 'PAM']

    if args.processNum == None:
        data = get_features(proteasesOne, fasta_dict, coding_scheme, True,
                            allmatrix,length)  
    else:


        p1 = Pool(args.processNum)
        result1 = p1.starmap(get_features1, [
            (proteasesOne, fasta_dict, 'WLS', True, allmatrix,length),
            (proteasesOne, fasta_dict, 'IC50', True, allmatrix,length),
            (proteasesOne, fasta_dict, 'AAF', True, allmatrix,length),
            (proteasesOne, fasta_dict, 'PWM', True, allmatrix,length),
            (proteasesOne, fasta_dict, 'PPM', True, allmatrix,length),
            (proteasesOne, fasta_dict, 'NNS', True, allmatrix,length),
            (proteasesOne, fasta_dict, 'KNN', True, allmatrix,length),
            (proteasesOne, fasta_dict, 'BSI', True, allmatrix,length),
            (proteasesOne, fasta_dict, 'PAM', True, allmatrix,length)
            ])
        p1.close()
        p1.join()

        data = pd.concat(
            [result1[0][0], result1[1][0], result1[2][0], result1[3][0], result1[4][0],
                result1[5][0], result1[6][0], result1[7][0], result1[8][0]],
            axis=1)

    return data

def get_features(proteaseName, trainpeptide,coding_scheme,pre,matrix,length):
    
    # length=8
    if pre:
        featuresDF=pd.DataFrame()
        for codS in coding_scheme:
            if codS =='WLS':
                featueDic=WLS(trainpeptide,length, None,matrix)
                WLS_f=[v[1] for v in featueDic['###']['pre']]
                featuresDF['WLS']=WLS_f
            elif codS =='IC50':
                featueDic=IC50(trainpeptide,length, None,matrix)
                IC50_f=[v[1] for v in featueDic['###']['pre']]
                featuresDF['IC50']=IC50_f
            elif codS == 'AAF':
                featueDic=AAF(trainpeptide,length, None,matrix)
                AAF_f=[v[1] for v in featueDic['###']['pre']]
                featuresDF['AAF']=AAF_f
            elif codS == 'PWM':
                featueDic=PWM(trainpeptide,length, None,matrix)
                PWM_f=[v[1] for v in featueDic['###']['pre']]          
                featuresDF['PWM']=PWM_f
            elif codS == 'BSI':
                for bls in blosumDict.keys():
                    featueDic=BSI(trainpeptide, None,blosumDict[bls],matrix,bls)
                    BSI_f=[v[1] for v in featueDic['###']['pre']]
                    featuresDF[bls]=BSI_f 
            elif codS == 'PAM':
                for pam in PAMDict.keys():
                    featueDic=PAM(trainpeptide,None ,PAMDict[pam],matrix,pam)
                    PAM_f=[v[1] for v in featueDic['###']['pre']]       
                    featuresDF[pam]=PAM_f 
            elif codS == 'KNN':
                featueDic= KNN(trainpeptide,length, None,matrix)
                KNN_f=[v[1] for v in featueDic['###']['pre']]   
                featuresDF['KNN']=KNN_f
            elif codS == 'NNS':
                featueDic= NNS(trainpeptide,None, matrix)
                NNS_f=[v[1] for v in featueDic['###']['pre']]        
                featuresDF['NNS']=NNS_f
                featueDic= PPM(trainpeptide,length, None,matrix)
                PPM_f=[v[1] for v in featueDic['###']['pre']]          
                featuresDF['PPM']=PPM_f               
            else:
                print(f'There is no feature descriptor {codS}')
        return featuresDF
    else:
        Count=Counter(trainpeptide.values())
        labels=[1]*int(Count['1']) + [0]*int(Count['0'])
        featuresDF=pd.DataFrame()#
        featuresDF['true_label']=labels
        allmatrix = OrderedDict()
        allmatrix[proteaseName] = OrderedDict()
        for codS in coding_scheme:
            if codS =='WLS':
                featueDic,WLSmatrix=WLS(None,length, trainpeptide,None)
                WLS_f=[v[1] for v in featueDic['pos']]
                WLS_f_=[v[1] for v in featueDic['neg']['0']]
                WLS_f.extend(WLS_f_)
                featuresDF['WLS']=WLS_f
                allmatrix[proteaseName][codS]=WLSmatrix[codS]
            elif codS =='IC50':
                featueDic,IC50matrix=IC50(None,length, trainpeptide,None)
                IC50_f=[v[1] for v in featueDic['pos']]
                IC50_f_=[v[1] for v in featueDic['neg']['0']]
                IC50_f.extend(IC50_f_)
                featuresDF['IC50']=IC50_f
                allmatrix[proteaseName][codS]=IC50matrix[codS]
            elif codS == 'AAF':
                featueDic,AAFmatrix=AAF(None,length, trainpeptide,None)
                AAF_f=[v[1] for v in featueDic['pos']]
                AAF_f_=[v[1] for v in featueDic['neg']['0']]
                AAF_f.extend(AAF_f_)
                featuresDF['AAF']=AAF_f
                allmatrix[proteaseName][codS]=AAFmatrix[codS]
            elif codS == 'PWM':
                featueDic,PWMmatrix=PWM(None,length, trainpeptide,None)
                PWM_f=[v[1] for v in featueDic['pos']]
                PWM_f_=[v[1] for v in featueDic['neg']['0']]
                PWM_f.extend(PWM_f_)            
                featuresDF['PWM']=PWM_f
                allmatrix[proteaseName][codS]=PWMmatrix[codS]
            elif codS == 'BSI':
                for bls in blosumDict.keys():
                    featueDic,BSImatrix=BSI(None, trainpeptide,blosumDict[bls],None,bls)
                    BSI_f=[v[1] for v in featueDic['pos']]
                    BSI_f_=[v[1] for v in featueDic['neg']['0']]
                    BSI_f.extend(BSI_f_)            
                    featuresDF[bls]=BSI_f 
                    allmatrix[proteaseName][bls]=BSImatrix[bls] 
            elif codS == 'PAM':
                for pam in PAMDict.keys():
                    featueDic,PAMmatrix=PAM(None, trainpeptide,PAMDict[pam],None,pam)
                    PAM_f=[v[1] for v in featueDic['pos']]
                    PAM_f_=[v[1] for v in featueDic['neg']['0']]
                    PAM_f.extend(PAM_f_)         
                    featuresDF[pam]=PAM_f 
                    allmatrix[proteaseName][pam]=PAMmatrix[pam] 
            elif codS == 'KNN':
                featueDic,KNNmatrix= KNN(None,length, trainpeptide,None)
                KNN_f=[v[1] for v in featueDic['pos']]
                KNN_f_=[v[1] for v in featueDic['neg']['0']]
                KNN_f.extend(KNN_f_)      
                featuresDF['KNN']=KNN_f
                allmatrix[proteaseName][codS]=KNNmatrix[codS]
            elif codS == 'NNS':
                featueDic,NNSmatrix= NNS(None, trainpeptide,None)
                NNS_f=[v[1] for v in featueDic['pos']]
                NNS_f_=[v[1] for v in featueDic['neg']['0']]
                NNS_f.extend(NNS_f_)          
                featuresDF['NNS']=NNS_f
                allmatrix[proteaseName][codS]=NNSmatrix[codS]   
            elif codS == 'PPM':
                featueDic,PPMmatrix= PPM(None,length, trainpeptide,None)
                PPM_f=[v[1] for v in featueDic['pos']]
                PPM_f_=[v[1] for v in featueDic['neg']['0']]
                PPM_f.extend(PPM_f_)        
                featuresDF['PPM']=PPM_f
                allmatrix[proteaseName][codS]=PPMmatrix[codS]                  
            else:
                print(f'There is no feature descriptor {codS}')
        return featuresDF,allmatrix

def get_features1(proteaseName, trainpeptide,codS,pre,matrix,length):
    # length=8
    if pre:
        featuresDF=pd.DataFrame()
        if codS =='WLS':
            featueDic=WLS(trainpeptide,length, None,matrix)
            WLS_f=[v[1] for v in featueDic['###']['pre']]
            featuresDF['WLS']=WLS_f
        elif codS =='IC50':
            featueDic=IC50(trainpeptide,length, None,matrix)
            IC50_f=[v[1] for v in featueDic['###']['pre']]
            featuresDF['IC50']=IC50_f
        elif codS == 'AAF':
            featueDic=AAF(trainpeptide,length, None,matrix)
            AAF_f=[v[1] for v in featueDic['###']['pre']]
            featuresDF['AAF']=AAF_f
        elif codS == 'PWM':
            featueDic=PWM(trainpeptide,length, None,matrix)
            PWM_f=[v[1] for v in featueDic['###']['pre']]           
            featuresDF['PWM']=PWM_f
        elif codS == 'BSI':
            for bls in blosumDict.keys():
                featueDic=BSI(trainpeptide, None,blosumDict[bls],matrix,bls)
                BSI_f=[v[1] for v in featueDic['###']['pre']]           
                featuresDF[bls]=BSI_f 
        elif codS == 'PAM':
            for pam in PAMDict.keys():
                featueDic=PAM(trainpeptide,None ,PAMDict[pam],matrix,pam)
                PAM_f=[v[1] for v in featueDic['###']['pre']]           
                featuresDF[pam]=PAM_f 
        elif codS == 'KNN':
            featueDic= KNN(trainpeptide,length, None,matrix)
            KNN_f=[v[1] for v in featueDic['###']['pre']]       
            featuresDF['KNN']=KNN_f
        elif codS == 'NNS':
            featueDic= NNS(trainpeptide,None, matrix)
            NNS_f=[v[1] for v in featueDic['###']['pre']]         
            featuresDF['NNS']=NNS_f 
        elif codS == 'PPM':
            featueDic= PPM(trainpeptide,length, None,matrix)
            PPM_f=[v[1] for v in featueDic['###']['pre']]           
            featuresDF['PPM']=PPM_f
        else:
            print(f'There is no feature descriptor {codS}')
        
        return featuresDF,'_'
    else:

        featuresDF=pd.DataFrame()
        allmatrix = {}
        allmatrix[proteaseName] = {}
        if codS =='WLS':
            featueDic,WLSmatrix=WLS(None,length, trainpeptide,None)
            WLS_f=[v[1] for v in featueDic['pos']]
            WLS_f_=[v[1] for v in featueDic['neg']['0']]
            WLS_f.extend(WLS_f_)

            featuresDF['WLS']=WLS_f
            allmatrix[proteaseName][codS]=WLSmatrix[codS]
            
        elif codS =='IC50':
            featueDic,IC50matrix=IC50(None,length, trainpeptide,None)
            IC50_f=[v[1] for v in featueDic['pos']]
            IC50_f_=[v[1] for v in featueDic['neg']['0']]
            IC50_f.extend(IC50_f_)
            featuresDF['IC50']=IC50_f

            allmatrix[proteaseName][codS]=IC50matrix[codS]
        elif codS == 'AAF':
            featueDic,AAFmatrix=AAF(None,length, trainpeptide,None)
            AAF_f=[v[1] for v in featueDic['pos']]
            AAF_f_=[v[1] for v in featueDic['neg']['0']]
            AAF_f.extend(AAF_f_)
            featuresDF['AAF']=AAF_f
            allmatrix[proteaseName][codS]=AAFmatrix[codS]
        elif codS == 'PWM':
            featueDic,PWMmatrix=PWM(None,length, trainpeptide,None)
            PWM_f=[v[1] for v in featueDic['pos']]
            PWM_f_=[v[1] for v in featueDic['neg']['0']]
            PWM_f.extend(PWM_f_)            
            featuresDF['PWM']=PWM_f
            allmatrix[proteaseName][codS]=PWMmatrix[codS]
        elif codS == 'BSI':
            for bls in blosumDict.keys():
                featueDic,BSImatrix=BSI(None, trainpeptide,blosumDict[bls],None,bls)
                BSI_f=[v[1] for v in featueDic['pos']]
                BSI_f_=[v[1] for v in featueDic['neg']['0']]
                BSI_f.extend(BSI_f_)            
                featuresDF[bls]=BSI_f 
                allmatrix[proteaseName][bls]=BSImatrix[bls] 
        elif codS == 'PAM':
            for pam in PAMDict.keys():
                featueDic,PAMmatrix=PAM(None, trainpeptide,PAMDict[pam],None,pam)
                PAM_f=[v[1] for v in featueDic['pos']]
                PAM_f_=[v[1] for v in featueDic['neg']['0']]
                PAM_f.extend(PAM_f_)
                featuresDF[pam]=PAM_f 
                allmatrix[proteaseName][pam]=PAMmatrix[pam] 
        elif codS == 'KNN':
            featueDic,KNNmatrix= KNN(None,length, trainpeptide,None)
            KNN_f=[v[1] for v in featueDic['pos']]
            KNN_f_=[v[1] for v in featueDic['neg']['0']]
            KNN_f.extend(KNN_f_) 
            featuresDF['KNN']=KNN_f
            allmatrix[proteaseName][codS]=KNNmatrix[codS]
        elif codS == 'NNS':
            featueDic,NNSmatrix= NNS(None, trainpeptide,None)
            NNS_f=[v[1] for v in featueDic['pos']]
            NNS_f_=[v[1] for v in featueDic['neg']['0']]
            NNS_f.extend(NNS_f_)
            featuresDF['NNS']=NNS_f
            allmatrix[proteaseName][codS]=NNSmatrix[codS]   
        elif codS == 'PPM':
            featueDic,PPMmatrix= PPM(None,length, trainpeptide,None)
            PPM_f=[v[1] for v in featueDic['pos']]
            PPM_f_=[v[1] for v in featueDic['neg']['0']]
            PPM_f.extend(PPM_f_)
            featuresDF['PPM']=PPM_f
            allmatrix[proteaseName][codS]=PPMmatrix[codS]           
        else:
            print(f'There is no feature descriptor {codS}')

        return featuresDF,allmatrix
