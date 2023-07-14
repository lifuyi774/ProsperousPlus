import pandas as pd
import os
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef

def calper(trainperformancefile, templatefolder):
    currentpath = os.getcwd()
    if trainperformancefile != None:
        filename = 'trainresult.txt'
    else:
        filename = 'testresult.txt'

    with open('%s/%s' % ( templatefolder, filename), 'r') as f:
        checkline = 0
        # labelscorelist = []
        truelabels,predlabels,predscores=[],[],[]
        for eachline in f:
            if 'inst#' in eachline:
                checkline += 1
            elif checkline == 1 and len(eachline) > 1 :
                labelcontentL = eachline.split(':')
                labelcontent=labelcontentL[1]
                predLabelScorecontent=eachline.split(' ')
                predscores.append(float(predLabelScorecontent[-2]))
                truelabels.append(int(labelcontent.split()[0]))
                predlabels.append(int(labelcontentL[2].split()[0]))

    scores_reverse=[]
    for i,v in enumerate(predlabels):
        if int(v) ==1:
            scores_reverse.append(predscores[i])
        elif int(v)==0:
            scores_reverse.append(1-float(predscores[i]))
    dict_performanceCV = OrderedDict()
    dict_performanceCV['Precision'] = round(precision_score(truelabels,predlabels), 3)
    dict_performanceCV['Recall'] = round(recall_score(truelabels,predlabels), 3)
    dict_performanceCV['F1 score'] = round(f1_score(truelabels,predlabels), 3)
    dict_performanceCV["MCC"] = round(matthews_corrcoef(truelabels,predlabels), 3)
    dict_performanceCV['Accuracy'] = round(accuracy_score(truelabels,predlabels),3)
    dict_performanceCV['AUC'] = round(roc_auc_score(truelabels,scores_reverse), 3)
    
    return dict_performanceCV,predlabels


def AODE_trainAndPred(args,codetype,protease,CV,x_train,y_train,x_test,y_test,rankA):

    currentpath = os.getcwd()
    print(currentpath)
    templatefolder2=args.outputpath
    p=str(len(x_train.columns))
    if codetype=="all":
        writeArrff(templatefolder2,'train',x_train,y_train,'Yes')
        filelist = os.listdir('%s' % (templatefolder2))
        os.chdir('source/weka-3-9-3')
        if CV=='LOO':
            CVtime='LOO'
        else:
            CVtime=CV
        for eachfile in filelist:
            if 'train.arff' in eachfile:
                modelname = protease+'_AODE_model'
                if CVtime != 'LOO':
                    # os.system('java weka.classifiers.meta.FilteredClassifier -x %s -F weka.filters.supervised.attribute.Discretize -W weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -d %s/%s/%s -p %s -t %s/%s/%s > %s/%s/trainresult.txt' % (CVtime, currentpath, templatefolder2, modelname,p,currentpath, templatefolder2, eachfile, currentpath, templatefolder2))
                    # # os.system('java weka.classifiers.meta.FilteredClassifier -x %s -F weka.filters.supervised.attribute.Discretize -W weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -d %s/%s -p %s -t %s/%s > %s/trainresult.txt' % (CVtime, args.outputpath, modelname,p,args.outputpath, eachfile, args.outputpath))
                    os.system(
                        'java weka.classifiers.meta.FilteredClassifier -x %s -F weka.filters.supervised.attribute.Discretize -W weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -d %s/%s -p %s -t %s/%s > %s/trainresult.txt' % (
                            CVtime, templatefolder2, modelname, p, templatefolder2, eachfile,
                            templatefolder2))
                else:
                    print(str(len(x_train)))
                    # os.system('java weka.classifiers.meta.FilteredClassifier -x %s -F weka.filters.supervised.attribute.Discretize -W weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -d %s/%s/%s -p %s -t %s/%s/%s > %s/%s/trainresult.txt' % (str(len(x_train)), currentpath, templatefolder2, modelname, p,currentpath, templatefolder2, eachfile, currentpath, templatefolder2))
                    os.system(
                        'java weka.classifiers.meta.FilteredClassifier -x %s -F weka.filters.supervised.attribute.Discretize -W weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -d %s/%s -p %s -t %s/%s > %s/trainresult.txt' % (
                            str(len(x_train)), templatefolder2, modelname, p, templatefolder2,
                            eachfile, templatefolder2))
                    # os.system('java weka.classifiers.meta.FilteredClassifier -x %s -F weka.filters.supervised.attribute.Discretize -W weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -d %s/%s -p %s -t %s/%s > %s/trainresult.txt' % (str(len(x_train)), args.outputpath, modelname, p,args.outputpath, eachfile, args.outputpath))
        os.chdir(os.path.pardir)
        os.chdir(os.path.pardir)
        # aode test
        p=str(len(x_train.columns))
        writeArrff(templatefolder2,'test',x_test,y_test,'Yes')
        filelist = os.listdir('%s' % (templatefolder2))
        os.chdir('source/weka-3-9-3') 
        for eachfile in filelist:
            if 'test.arff' in eachfile:
                resultname = 'testresult.txt'
                attributenum = p
                # os.system('java weka.classifiers.meta.FilteredClassifier -p %s -l %s/%s/%s -T %s/%s/%s > %s/%s/%s' % (str(attributenum), currentpath, templatefolder2,  modelname, currentpath, templatefolder2, eachfile, currentpath, templatefolder2, resultname))
                # # os.system('java weka.classifiers.meta.FilteredClassifier -p %s -l %s/%s -T %s/%s > %s/%s' % (str(attributenum), args.outputpath,  modelname, args.outputpath, eachfile, args.outputpath, resultname))
                os.system('java weka.classifiers.meta.FilteredClassifier -p %s -l %s/%s -T %s/%s > %s/%s' % (
                    str(attributenum), templatefolder2, modelname, templatefolder2, eachfile,
                    templatefolder2, resultname))

    os.chdir(os.path.pardir)
    os.chdir(os.path.pardir)
    # cal aode perfomance
    rankN=rankA
    dict_trainperformanceCV,aodeCVpredLabel = calper('yes', templatefolder2)
    aode_metrics_DF=pd.DataFrame()
    aode_metrics_DF['model_rank']=[rankN]
    aode_metrics_DF['Model']=['AODE']
    aode_metrics_DF['Accuracy']=[dict_trainperformanceCV ['Accuracy']]
    aode_metrics_DF['AUC']=[dict_trainperformanceCV['AUC']]
    aode_metrics_DF['Recall']=[dict_trainperformanceCV['Recall'] ]
    aode_metrics_DF['Prec.']=[dict_trainperformanceCV['Precision'] ]
    aode_metrics_DF['F1']=[dict_trainperformanceCV['F1 score'] ]
    aode_metrics_DF['MCC']=[dict_trainperformanceCV['MCC'] ]
    
    dict_testperformanceCV,aode_testset_predLabel = calper(None, templatefolder2)
    aode_test_preP=pd.DataFrame()
    aode_test_preP['Model']=['AODE']
    aode_test_preP['Model_name']=['AODE']
    aode_test_preP['Accuracy']=[dict_testperformanceCV['Accuracy']]
    aode_test_preP['AUC']=[dict_testperformanceCV['AUC']]
    aode_test_preP['Recall']=[dict_testperformanceCV['Recall'] ]
    aode_test_preP['Prec.']=[dict_testperformanceCV['Precision'] ]
    aode_test_preP['F1']=[dict_testperformanceCV['F1 score'] ]
    aode_test_preP['MCC']=[dict_testperformanceCV['MCC'] ]

    return aode_metrics_DF,aode_test_preP,aode_testset_predLabel,modelname

def AODE_train(args,codetype,protease,CV,x_train,y_train,rankA):
    templatefolder2=args.outputpath

    p=str(len(x_train.columns))
 
    if codetype=="all":
        writeArrff(templatefolder2,'train',x_train,y_train,'Yes')

        filelist = os.listdir('%s' % (templatefolder2))

        os.chdir('source/weka-3-9-3')
        if CV=='LOO':
            CVtime='LOO'
        else:
            CVtime=CV
        for eachfile in filelist:
            if 'train.arff' in eachfile:
                modelname = protease+'_AODE_model'
                if CVtime != 'LOO':
                    # os.system('java weka.classifiers.meta.FilteredClassifier -x %s -F weka.filters.supervised.attribute.Discretize -W weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -d %s/%s/%s -p %s -t %s/%s/%s > %s/%s/trainresult.txt' % (CVtime, currentpath, templatefolder2, modelname,p,currentpath, templatefolder2, eachfile, currentpath, templatefolder2))
                    # # os.system('java weka.classifiers.meta.FilteredClassifier -x %s -F weka.filters.supervised.attribute.Discretize -W weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -d %s/%s -p %s -t %s/%s > %s/trainresult.txt' % (CVtime, args.outputpath, modelname,p,args.outputpath, eachfile, args.outputpath))
                    os.system(
                        'java weka.classifiers.meta.FilteredClassifier -x %s -F weka.filters.supervised.attribute.Discretize -W weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -d %s/%s -p %s -t %s/%s > %s/trainresult.txt' % (
                            CVtime, templatefolder2, modelname, p, templatefolder2, eachfile,
                            templatefolder2))
                else:
                    print(str(len(x_train)))
                    # os.system('java weka.classifiers.meta.FilteredClassifier -x %s -F weka.filters.supervised.attribute.Discretize -W weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -d %s/%s/%s -p %s -t %s/%s/%s > %s/%s/trainresult.txt' % (str(len(x_train)), currentpath, templatefolder2, modelname, p,currentpath, templatefolder2, eachfile, currentpath, templatefolder2))
                    os.system(
                        'java weka.classifiers.meta.FilteredClassifier -x %s -F weka.filters.supervised.attribute.Discretize -W weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -d %s/%s -p %s -t %s/%s > %s/trainresult.txt' % (
                            str(len(x_train)), templatefolder2, modelname, p, templatefolder2,
                            eachfile, templatefolder2))
                    # os.system('java weka.classifiers.meta.FilteredClassifier -x %s -F weka.filters.supervised.attribute.Discretize -W weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -d %s/%s -p %s -t %s/%s > %s/trainresult.txt' % (str(len(x_train)), args.outputpath, modelname, p,args.outputpath, eachfile, args.outputpath))
    os.chdir(os.path.pardir)
    os.chdir(os.path.pardir)

    rankN=rankA
    dict_trainperformanceCV,aodeCVpredLabel = calper('yes', templatefolder2)#
    aode_metrics_DF=pd.DataFrame()
    aode_metrics_DF['model_rank']=[rankN]
    aode_metrics_DF['Model']=['AODE']
    aode_metrics_DF['Accuracy']=[dict_trainperformanceCV ['Accuracy']]
    aode_metrics_DF['AUC']=[dict_trainperformanceCV['AUC']]
    aode_metrics_DF['Recall']=[dict_trainperformanceCV['Recall'] ]
    aode_metrics_DF['Prec.']=[dict_trainperformanceCV['Precision'] ]
    aode_metrics_DF['F1']=[dict_trainperformanceCV['F1 score'] ]
    aode_metrics_DF['MCC']=[dict_trainperformanceCV['MCC'] ]  
    return aode_metrics_DF

def writeArrff(pathdir,filename,data,labelY,HasLABEL):
    if HasLABEL =='Yes':
        with open('%s/%s.arff' % (pathdir,filename), 'w') as f:
        # with open('%s/train.arff' % (args.outputpath), 'w') as f:
            f.write('@RELATION protease\n')
            f.write('\n')
            for featurename in list(data.columns):   
                f.write('@ATTRIBUTE' + ' ' + featurename + ' ' + 'REAL\n')  
            f.write('@ATTRIBUTE' + ' ' + 'class' + ' ' + '{1, 0}\n')
            f.write('\n')
            f.write('@DATA\n') 
            for n,featureValues in enumerate(zip(data.values)):
                for i in list(featureValues[0])[0:]:
                    f.write(str(i) + ' ')
                # f.write(str(int(featureValues[0][0])))
                f.write(str(int(list(labelY.values)[n])))
                f.write('\n')
    else:
        with open('%s/%s.arff' % (pathdir,filename), 'w') as f:
            f.write('@RELATION peptide\n')
            f.write('\n')
            for featurename in list(data.columns):
                f.write('@ATTRIBUTE' + ' ' + featurename + ' ' + 'REAL\n')
            f.write('@ATTRIBUTE' + ' ' + 'class' + ' ' + '{1, 0}\n')
            f.write('\n')
            f.write('@DATA\n')
            # for eachseq, feas in pepfea.items():
            #     for eachfea in feas:
            #         f.write(str(eachfea) + ' ')
            #     f.write('?')
            #     f.write('\n')
            for n, featureValues in enumerate(zip(data.values)):
                for i in list(featureValues[0])[0:]:
                    f.write(str(i) + ' ')
                f.write('?')
                f.write('\n')

def AODE_predict(args,AodeModelNa,predata):

    currentpath = os.getcwd()
    templatefolder3 = args.outputpath
    writeArrff(templatefolder3,'predict',predata,None,'No')

    filelist = os.listdir('%s' % (templatefolder3))
    p=str(len(predata.columns))
    modelfile=AodeModelNa
    dict_nameseqscore = OrderedDict()
    os.chdir('source/weka-3-9-3')
    for eachfile in filelist:
        if 'predict.arff' in eachfile:
            resultname = '%s_result.txt' % ('AODE')
            # os.system('java weka.classifiers.meta.FilteredClassifier -p 11 -l %s -T %s/%s/%s > %s/%s/%s' % (
            # args.modelfile, currentpath, templatefolder3, eachfile, currentpath, templatefolder3, resultname))
            os.system('java weka.classifiers.meta.FilteredClassifier -p %s -l %s -T %s/%s > %s/%s' % (p,
                modelfile, templatefolder3, eachfile, templatefolder3, resultname))
    os.chdir(os.path.pardir)
    os.chdir(os.path.pardir)
    # with open('%s/%s/%s' % (currentpath, templatefolder3, resultname), 'r') as f:
    with open('%s/%s' % (templatefolder3, resultname), 'r') as f:
        checkline = 0
        # labelscorelist = []
        predlabels, predscores = [], []
        for eachline in f:
            if 'inst#' in eachline:
                checkline += 1
            elif checkline == 1 and len(eachline) > 1:
                labelcontentL = eachline.split(':')
                labelcontent = labelcontentL[1]
                predLabelScorecontent = eachline.split(' ')
                predscores.append(float(predLabelScorecontent[-2]))
                predlabels.append(int(labelcontentL[2].split()[0]))
    scores_reverse=[]
    for i,v in enumerate(predlabels):
        if int(v) ==1:
            scores_reverse.append(predscores[i])
        elif int(v)==0:
            scores_reverse.append(round(1-float(predscores[i]),3))
    return predlabels, scores_reverse
