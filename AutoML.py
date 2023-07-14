import os
from pycaret.classification import setup,get_config,set_config,compare_models,predict_model,pull,create_model,blend_models,stack_models,ensemble_model,save_config,save_model,load_model,load_config,interpret_model
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef
import pymrmr
import copy
import redis
from collections import OrderedDict,Counter
import shutil
import pandas as pd
import numpy as np
from functools import reduce
from AODE import *
from myplot import *
from zip import zip_folder
from sklearn import preprocessing 
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.getOrCreate()
# spark.sparkContext.setLogLevel("WARN")
# from pycaret.parallel import FugueBackend

def AutoML(args,selectedModels,n_select,templatefolder,x_train,y_train,x_test,y_test,predata,fold_CV,codetype,MLs,allmatrixName,predictfasta_dict,predict_seqname,TrSEQLENGTH):
    # compare models
    compare_selectedModels=compare_models(include=selectedModels,n_select=n_select,sort='Accuracy',probability_threshold=0.5) #,parallel=FugueBackend(spark)
    pull_compare_selectedModels=pull()
    print(pull_compare_selectedModels.head())
    # top3 base model name
    top3baseModelsName=list(pull_compare_selectedModels.index)[:3] 
    
    # mRMR
    mRMR_data = pd.concat([y_train,x_train], axis=1)
    NUM=mRMR_data.shape[1]
    mrmr_output=pymrmr.mRMR(mRMR_data, 'MID',(NUM-1)) 
    mrmr_file=templatefolder+'/'+args.protease+'_selected_feature_list_MID.txt'
    mrmr_write = open(mrmr_file, 'a')
    for i in range(len(mrmr_output)):
        s = str(mrmr_output[i]).replace('[', '').replace(']', '')  
        s = s.replace("'", '').replace(',', '') + '\n'  
        mrmr_write.write(s)
    mrmr_write.close()
    read_mrmr = pd.read_csv( mrmr_file, header=None)
    mrmr = [list(v)[0] for v in list(read_mrmr.values)]

    # IFS  
    for i in range(1,NUM):
        x_train_new = pd.concat([x_train.loc[:, mrmr[:i]], y_train], axis=1)
        x_test_new = pd.concat([x_test.loc[:, mrmr[:i]], y_test], axis=1)

        numeric_features=[]
        for col in x_train_new.columns:
            if 'KNN' in col:
                numeric_features.append(col)

        exp1=setup(data=x_train_new,target='true_label',test_data=x_test_new,session_id=123,use_gpu=False,normalize=False,remove_perfect_collinearity=False,fold=fold_CV,silent=True,verbose=False,preprocess=False,numeric_features=numeric_features)
        
        # print(get_config('X_train').shape)
        baseModel_0 = create_model(top3baseModelsName[0])
        result_baseModel_0 = pull()
        result_baseModel_0 = result_baseModel_0[fold_CV:fold_CV+1] # CV mean
        result_baseModel_0.insert(0, 'num', i)
        result_baseModel_0.to_csv("{}/CV_IFS_combined_results_train_matrix_MID_{}.txt".format(templatefolder,top3baseModelsName[0]), mode='a', header=False)
        baseModel_1 = create_model(top3baseModelsName[1])
        result_baseModel_1 = pull()
        result_baseModel_1 = result_baseModel_1[fold_CV:fold_CV+1]
        result_baseModel_1.insert(0, 'num', i)
        result_baseModel_1.to_csv("{}/CV_IFS_combined_results_train_matrix_MID_{}.txt".format(templatefolder,top3baseModelsName[1]), mode='a', header=False)
        baseModel_2 = create_model(top3baseModelsName[2])
        result_baseModel_2 = pull()
        result_baseModel_2 = result_baseModel_2[fold_CV:fold_CV+1]
        result_baseModel_2.insert(0, 'num', i)
        result_baseModel_2.to_csv("{}/CV_IFS_combined_results_train_matrix_MID_{}.txt".format(templatefolder,top3baseModelsName[2]), mode='a', header=False)
        if 'AODE' in MLs:
            IFS_result_aode=AODE_train(args,codetype,args.protease,fold_CV,x_train.loc[:, mrmr[:i]],y_train,8)
            IFS_result_aode.to_csv("{}/CV_IFS_combined_results_train_matrix_MID_{}.txt".format(templatefolder,'AODE'), mode='a', header=False)

    # select features
    IFS_num = pd.read_csv("{}/CV_IFS_combined_results_train_matrix_MID_{}.txt".format(templatefolder,top3baseModelsName[0]),header = None)
    ACC,AUC = IFS_num.iloc[:,2],IFS_num.iloc[:,3]
    top_point = list(zip(np.arange(1,NUM),ACC,AUC))
    Num_features_baseModel_0 = sorted(top_point, key = lambda kv:(kv[1], kv[2]),reverse = True)[0][0]
    
    IFS_num = pd.read_csv("{}/CV_IFS_combined_results_train_matrix_MID_{}.txt".format(templatefolder,top3baseModelsName[1]),header = None)
    ACC,AUC = IFS_num.iloc[:,2],IFS_num.iloc[:,3]
    top_point = list(zip(np.arange(1,NUM),ACC,AUC))
    Num_features_baseModel_1 = sorted(top_point, key = lambda kv:(kv[1], kv[2]),reverse = True)[0][0]
    
    IFS_num = pd.read_csv("{}/CV_IFS_combined_results_train_matrix_MID_{}.txt".format(templatefolder,top3baseModelsName[2]),header = None)
    ACC,AUC = IFS_num.iloc[:,2],IFS_num.iloc[:,3]
    top_point = list(zip(np.arange(1,NUM),ACC,AUC))
    Num_features_baseModel_2 = sorted(top_point, key = lambda kv:(kv[1], kv[2]),reverse = True)[0][0]  
    if 'AODE' in MLs:
        IFS_num = pd.read_csv("{}/CV_IFS_combined_results_train_matrix_MID_{}.txt".format(templatefolder,'AODE'),header = None)
        ACC,AUC = IFS_num.iloc[:,3],IFS_num.iloc[:,4] # 
        top_point = list(zip(np.arange(1,NUM),ACC,AUC))
        Num_features_baseModel_aode = sorted(top_point, key = lambda kv:(kv[1], kv[2]),reverse = True)[0][0]
  
    baseModel_0_FETURES = mrmr[:Num_features_baseModel_0]
    baseModel_1_FETURES = mrmr[:Num_features_baseModel_1]
    baseModel_2_FETURES = mrmr[:Num_features_baseModel_2]
    CAT_FEATURES = [baseModel_0_FETURES,baseModel_1_FETURES,baseModel_2_FETURES]
    Ensemble_FEATURES = list(reduce(lambda x,y : set(x) | set(y), CAT_FEATURES ))
    # zip files
    file_baseModel_0='0_baseModel_results'
    os.mkdir('%s/%s' % (templatefolder, file_baseModel_0))
    np.save('{}/{}/featuresSorted.npy'.format(templatefolder,file_baseModel_0),baseModel_0_FETURES)
    file_baseModel_1='1_baseModel_results'
    os.mkdir('%s/%s' % (templatefolder, file_baseModel_1))
    np.save('{}/{}/featuresSorted.npy'.format(templatefolder,file_baseModel_1),baseModel_1_FETURES)
    file_baseModel_2='2_baseModel_results'
    os.mkdir('%s/%s' % (templatefolder, file_baseModel_2))
    np.save('{}/{}/featuresSorted.npy'.format(templatefolder,file_baseModel_2),baseModel_2_FETURES)
    np.save('{}/featuresSorted_Ens.npy'.format(templatefolder),Ensemble_FEATURES)
    
    train_X = x_train
    train_y = y_train
    test_X = x_test
    test_y = y_test
    
    # model1 pipeline
    x_train_new = pd.concat([x_train.loc[:, baseModel_0_FETURES], y_train], axis=1)
    x_test_new = pd.concat([x_test.loc[:, baseModel_0_FETURES], y_test], axis=1)
    numeric_features=[]
    for col in x_train_new.columns:
        if 'KNN' in col:
            numeric_features.append(col)

    # setup
    exp11=setup(data=x_train_new,target='true_label',test_data=x_test_new,session_id=123,use_gpu=False,normalize=False,remove_perfect_collinearity=False,fold=fold_CV,silent=True,verbose=False,preprocess=False,numeric_features=numeric_features)
    # create model
    baseModel_0_with_OptFEATURES = create_model(top3baseModelsName[0])
    CVresult_bM_0_wOptF=pull()
    CVresult_bM_0_wOptF =CVresult_bM_0_wOptF[fold_CV:fold_CV+1]
    CVresult_bM_0_wOptF.insert(0,'Model',top3baseModelsName[0])
    CVresult_bM_0_wOptF.insert(0,'model_rank',0)
    CVresult_bM_0_wOptF=CVresult_bM_0_wOptF.drop('Kappa',1)
    save_model(baseModel_0_with_OptFEATURES,templatefolder+'/'+file_baseModel_0+'/'+top3baseModelsName[0]+'_MLmodel')
    
    # model2 pipeline
    x_train_new1 = pd.concat([x_train.loc[:, baseModel_1_FETURES], y_train], axis=1)
    x_test_new1 = pd.concat([x_test.loc[:, baseModel_1_FETURES], y_test], axis=1)
    numeric_features=[]
    for col in x_train_new1.columns:
        if 'KNN' in col:
            numeric_features.append(col)
            
    exp12=setup(data=x_train_new1,target='true_label',test_data=x_test_new1,session_id=123,use_gpu=False,normalize=False,remove_perfect_collinearity=False,fold=fold_CV,silent=True,verbose=False,preprocess=False,numeric_features=numeric_features)
    
    baseModel_1_with_OptFEATURES = create_model(top3baseModelsName[1])
    CVresult_bM_1_wOptF=pull()
    CVresult_bM_1_wOptF =CVresult_bM_1_wOptF[fold_CV:fold_CV+1]
    CVresult_bM_1_wOptF.insert(0,'Model',top3baseModelsName[1])
    CVresult_bM_1_wOptF.insert(0,'model_rank',1)
    CVresult_bM_1_wOptF=CVresult_bM_1_wOptF.drop('Kappa',1)
    save_model(baseModel_1_with_OptFEATURES,templatefolder+'/'+file_baseModel_1+'/'+top3baseModelsName[1]+'_MLmodel')
    
    # model3 pipeline
    x_train_new2 = pd.concat([x_train.loc[:, baseModel_2_FETURES], y_train], axis=1)
    x_test_new2 = pd.concat([x_test.loc[:, baseModel_2_FETURES], y_test], axis=1)
    numeric_features=[]
    for col in x_train_new2.columns:
        if 'KNN' in col:
            numeric_features.append(col)
            
    exp13=setup(data=x_train_new2,target='true_label',test_data=x_test_new2,session_id=123,use_gpu=False,normalize=False,remove_perfect_collinearity=False,fold=fold_CV,silent=True,verbose=False,preprocess=False,numeric_features=numeric_features) 
    
    baseModel_2_with_OptFEATURES = create_model(top3baseModelsName[2])
    CVresult_bM_2_wOptF=pull()
    CVresult_bM_2_wOptF =CVresult_bM_2_wOptF[fold_CV:fold_CV+1]
    CVresult_bM_2_wOptF.insert(0,'Model',top3baseModelsName[2])
    CVresult_bM_2_wOptF.insert(0,'model_rank',2)
    CVresult_bM_2_wOptF=CVresult_bM_2_wOptF.drop('Kappa',1)
    save_model(baseModel_2_with_OptFEATURES,templatefolder+'/'+file_baseModel_2+'/'+top3baseModelsName[2]+'_MLmodel')

    # ensemble model pipeline
    x_train_new3 = pd.concat([x_train.loc[:, Ensemble_FEATURES], y_train], axis=1)
    x_test_new3 = pd.concat([x_test.loc[:, Ensemble_FEATURES], y_test], axis=1)
    numeric_features=[]
    for col in x_train_new3.columns:
        if 'KNN' in col:
            numeric_features.append(col)

            
    exp14=setup(data=x_train_new3,target='true_label',test_data=x_test_new3,session_id=123,use_gpu=False,normalize=False,remove_perfect_collinearity=False,fold=fold_CV,silent=True,verbose=False,preprocess=False,numeric_features=numeric_features)
    
    estimator_list=[baseModel_0_with_OptFEATURES,baseModel_1_with_OptFEATURES,baseModel_2_with_OptFEATURES]
    #stack
    stacker = stack_models(estimator_list = [baseModel_0_with_OptFEATURES,baseModel_1_with_OptFEATURES,baseModel_2_with_OptFEATURES]) 
    results_stacker=pull()  
    results_stacker = results_stacker[fold_CV:fold_CV+1]
    stackername='_'.join(top3baseModelsName)+'_stacking' 
    results_stacker.insert(0,'Model',stackername)
    results_stacker.insert(0,'model_rank',3)
    results_stacker=results_stacker.drop('Kappa',1)
    results_stacker.head()
    save_model(stacker,templatefolder+'/'+stackername+'_MLmodel')
    #blend
    estimator_list_=copy.deepcopy(estimator_list)
    blenderyes=''
    for na in top3baseModelsName: 
        if na in ['svm']: 
            del estimator_list_[top3baseModelsName.index(na)]
    if len(estimator_list_)>=2:
        blender = blend_models(estimator_list = estimator_list_,method='soft')
        results_blender=pull()
        results_blender = results_blender[fold_CV:fold_CV+1]
        results_blender.head()
        blenderyes='yes'
    if blenderyes=='yes':
        blendername='_'.join(top3baseModelsName)+'_Voting'
        results_blender.insert(0,'Model',blendername)
        results_blender.insert(0,'model_rank',4)
        results_blender=results_blender.drop('Kappa',1)
        results_blender.head()
        save_model(stacker,templatefolder+'/'+blendername+'_MLmodel')
    if blenderyes=='yes':
        rnkbag=5
        if 'AODE' in MLs:
            rankA=8
    else:
        rnkbag=4
        if 'AODE' in MLs:
            rankA=7
    #bagging
    bagged_top3 = [] # 
    bagging_names=[]
    baggingDF=pd.DataFrame()
    for i,baseM in enumerate(estimator_list): 
        baggingmoedel=ensemble_model(baseM,verbose=False)
        bagging_pull=pull()
        results_bagging = bagging_pull[fold_CV:fold_CV+1]
        baggingname=top3baseModelsName[i]+'_bagging'
        bagging_names.append(baggingname)
        results_bagging.insert(0,'Model',baggingname)
        results_bagging.insert(0,'model_rank',i+rnkbag)
        results_bagging=results_bagging.drop('Kappa',1)
        baggingDF=baggingDF.append(results_bagging)
        bagged_top3.append(baggingmoedel)
        save_model(stacker,templatefolder+'/'+baggingname+'_MLmodel')
    #AODE
    if 'AODE' in MLs:
        aodeCVpred,aode_test_pred,aode_test_predLabel,AodeModelNa=AODE_trainAndPred(args,codetype,args.protease,fold_CV,x_train.loc[:, mrmr[:Num_features_baseModel_aode]],y_train,x_test.loc[:, mrmr[:Num_features_baseModel_aode]],y_test,rankA)
        # aode zip file
        baseModel_aode_FETURES = mrmr[:Num_features_baseModel_aode]
        file_baseModel_aode='{}_baseModel_results'.format(str(rankA))
        os.mkdir('%s/%s' % (templatefolder, file_baseModel_aode))
        np.save('{}/{}/featuresSorted.npy'.format(templatefolder,file_baseModel_aode),baseModel_aode_FETURES)
        np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_aode),TrSEQLENGTH) 
        
        shutil.move(templatefolder+'/'+AodeModelNa,templatefolder+'/'+file_baseModel_aode)
        shutil.copy(allmatrixName,templatefolder+'/'+file_baseModel_aode+'/'+'allmatrix.json')
        zip_folder(templatefolder+'/'+file_baseModel_aode,templatefolder+'/'+file_baseModel_aode+'.zip') 

    if blenderyes=='yes' and 'AODE' in MLs:
        alltrainedModel_pred_metric=pd.concat([CVresult_bM_0_wOptF,CVresult_bM_1_wOptF,CVresult_bM_2_wOptF,results_stacker,results_blender,baggingDF,aodeCVpred],ignore_index=True)
        alltrainedModel_pred_metric['indx']=top3baseModelsName+list(results_stacker['Model'].values)+list(results_blender['Model'].values)+list(baggingDF['Model'].values)+['AODE']
        alltrainedModel_pred_metric.set_index('indx',inplace=True)
    if blenderyes=='yes' and 'AODE' not in MLs:
        alltrainedModel_pred_metric=pd.concat([CVresult_bM_0_wOptF,CVresult_bM_1_wOptF,CVresult_bM_2_wOptF,results_stacker,results_blender,baggingDF],ignore_index=True)
        alltrainedModel_pred_metric['indx']=top3baseModelsName+list(results_stacker['Model'].values)+list(results_blender['Model'].values)+list(baggingDF['Model'].values)
        alltrainedModel_pred_metric.set_index('indx',inplace=True)
    elif blenderyes!='yes' and 'AODE'  in MLs:
        alltrainedModel_pred_metric=pd.concat([CVresult_bM_0_wOptF,CVresult_bM_1_wOptF,CVresult_bM_2_wOptF,results_stacker,baggingDF,aodeCVpred],ignore_index=True)
        alltrainedModel_pred_metric['indx']=top3baseModelsName+list(results_stacker['Model'].values)+list(baggingDF['Model'].values)+['AODE']
        alltrainedModel_pred_metric.set_index('indx',inplace=True)
    elif blenderyes!='yes' and 'AODE'  not in MLs:
        alltrainedModel_pred_metric=pd.concat([CVresult_bM_0_wOptF,CVresult_bM_1_wOptF,CVresult_bM_2_wOptF,results_stacker,baggingDF],ignore_index=True)
        alltrainedModel_pred_metric['indx']=top3baseModelsName+list(results_stacker['Model'].values)+list(baggingDF['Model'].values)
        alltrainedModel_pred_metric.set_index('indx',inplace=True)
        

    alltrainedModel_pred_metric.sort_values(by="Accuracy" , inplace=True, ascending=False) # AUC,Accuracy
    alltrainedModel_pred_metric.to_csv(templatefolder+'/allModels_CVresults.txt',index=False)
    top3Moldes_name=alltrainedModel_pred_metric['Model'].values[0:3] 
    
    if blenderyes=='yes':
        All_trained_models=estimator_list+[stacker]+[blender]+bagged_top3
    else:
        All_trained_models=estimator_list+[stacker]+bagged_top3
    
    top3_models_rank=alltrainedModel_pred_metric['model_rank'].values[0:3] 
      
    top3Models=[]
    
    for i in top3_models_rank:
        if 'AODE' in MLs:
            if int(i)==rankA:
                top3Models.append('AODE')
            else:
                top3Models.append(All_trained_models[i]) 
        else:
            top3Models.append(All_trained_models[i]) 

    if args.testfile:      
        labelTag=[]
        fprList=[]
        tprList=[]
        aucscore=[]
        checkpoint=0
        if 'AODE' not in MLs:
            rankA=''
        for i,m in enumerate(list(top3_models_rank)):

            if 'AODE' in MLs:
                if int(m)==rankA:
                    fpr,tpr, thresholds=roc_curve(y_test.values,aode_test_predLabel) 
                    rocTag='ROC of AODE'
                    auroc=auc(fpr,tpr)
                    fprList.append(fpr)
                    tprList.append(tpr)
                    aucscore.append(auroc)
                    labelTag.append(rocTag)
                    
                    if checkpoint==0:
                        top3pred_metrics=aode_test_pred
                        checkpoint=1
                    else:
                        top3pred_metrics=top3pred_metrics.append(aode_test_pred) 
            if int(m)==rankA and 'AODE' in MLs:
                continue
            else:
                if int(m)==0: # base model 0
                    loadedModel=templatefolder+'/'+file_baseModel_0+'/'+top3Moldes_name[i]+'_MLmodel'
                    clf=load_model(loadedModel)
                    set_config('X_test', test_X.loc[:, mrmr[:Num_features_baseModel_0]])
                elif int(m)==1:
                    loadedModel=templatefolder+'/'+file_baseModel_1+'/'+top3Moldes_name[i]+'_MLmodel'
                    clf=load_model(loadedModel)
                    set_config('X_test', test_X.loc[:, mrmr[:Num_features_baseModel_1]])
                elif int(m)==2:
                    loadedModel=templatefolder+'/'+file_baseModel_2+'/'+top3Moldes_name[i]+'_MLmodel'
                    clf=load_model(loadedModel)
                    set_config('X_test', test_X.loc[:, mrmr[:Num_features_baseModel_2]])
                elif int(m)!=rankA and int(m)>2: # ensemble model
                    loadedModel=templatefolder+'/'+top3Moldes_name[i]+'_MLmodel'
                    clf=load_model(loadedModel)
                    set_config('X_test', test_X.loc[:, Ensemble_FEATURES])
  
                # clf=load_model(loadedModel)
                # exp_for_test
                pred=predict_model(clf,raw_score=True)
                pred.to_csv(templatefolder+'/testdata_pre.txt')
                
                pred_metrics=pull()
                pred_metrics.insert(1,'Model_name',top3Moldes_name[i])
                pred_metrics=pred_metrics.drop(['Kappa'],1)
                if checkpoint==0:
                    top3pred_metrics=pred_metrics
                    checkpoint=1
                else:
                    top3pred_metrics=top3pred_metrics.append(pred_metrics)
                # ROC     
                if 'Score_1' in pred.columns:
                    pre_s=pred['Score_1']
                else:
                    pre_s=pred['Label']
                fpr,tpr, thresholds=roc_curve(pred['true_label'],pre_s) 
                rocTag='ROC of '+top3Moldes_name[i]
                auroc=auc(fpr,tpr)
                fprList.append(fpr)
                tprList.append(tpr)
                aucscore.append(auroc)
                labelTag.append(rocTag)
        
        # plot ROC
        plotAuc(fprList, tprList, aucscore, labelTag, False,templatefolder)
    if args.SHAP=='Yes':
        if int(top3_models_rank[0])!=rankA:
            if int(top3_models_rank[0])==0:
                test_SHAP=test_X.loc[:, mrmr[:Num_features_baseModel_0]]
                train_SHAP=train_X.loc[:, mrmr[:Num_features_baseModel_0]]
            elif int(top3_models_rank[0])==1:
                test_SHAP=test_X.loc[:, mrmr[:Num_features_baseModel_1]]
                train_SHAP=train_X.loc[:, mrmr[:Num_features_baseModel_1]]
            elif int(top3_models_rank[0])==2:
                test_SHAP=test_X.loc[:, mrmr[:Num_features_baseModel_2]]
                train_SHAP=train_X.loc[:, mrmr[:Num_features_baseModel_2]]
            elif int(top3_models_rank[0])!=rankA and int(top3_models_rank[0])>2:
                test_SHAP=test_X.loc[:, Ensemble_FEATURES]
                train_SHAP=train_X.loc[:, Ensemble_FEATURES]
            SHAP_plot(args,templatefolder,top3Moldes_name,top3Models,train_SHAP,test_SHAP) 
    

    if 'AODE' not in top3Moldes_name and 'AODE' in MLs:
        if codetype=='all':
            os.remove('%s/test.arff' % (templatefolder))
            os.remove('%s/train.arff' % (templatefolder))
            os.remove('%s/testresult.txt' % (templatefolder))
            os.remove('%s/trainresult.txt' % (templatefolder))
            os.remove('{}/{}.zip'.format(templatefolder, file_baseModel_aode))

    if args.testfile:
        finaltop3pred_metrics=top3pred_metrics.iloc[0:3,1:]
        finaltop3pred_metrics.to_csv(templatefolder+'/TestResults.txt',index=False)
    
    for idx,r in enumerate(top3_models_rank):
        if 'AODE' in MLs:
            if r==rankA:
                os.remove('%s/test.arff' % (templatefolder))
                os.remove('%s/train.arff' % (templatefolder))
                os.remove('%s/testresult.txt' % (templatefolder))
                os.remove('%s/trainresult.txt' % (templatefolder))
                if idx ==0:
                    os.rename(templatefolder+'/'+file_baseModel_aode+'.zip', templatefolder+'/0_Model')
                elif idx ==1:
                    os.rename(templatefolder+'/'+file_baseModel_aode+'.zip', templatefolder+'/1_Model')
                elif idx ==2:
                    os.rename(templatefolder+'/'+file_baseModel_aode+'.zip', templatefolder+'/2_Model')
            else:   
                if int(r) ==0: # base model 0
                    dir_model_name=templatefolder+'/'+file_baseModel_0
                    np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_0),TrSEQLENGTH)
                    shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')
                    if idx ==0:
                        zip_folder(dir_model_name,templatefolder+'/'+'0_Model')
                    elif idx ==1:
                        zip_folder(dir_model_name,templatefolder+'/'+'1_Model') 
                    elif idx ==2:
                        zip_folder(dir_model_name,templatefolder+'/'+'2_Model')
                elif int(r) ==1:
                    dir_model_name=templatefolder+'/'+file_baseModel_1 
                    np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_1),TrSEQLENGTH)
                    shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')
                    if idx ==0:
                        zip_folder(dir_model_name,templatefolder+'/'+'0_Model')    
                    elif idx ==1:
                        zip_folder(dir_model_name,templatefolder+'/'+'1_Model')
                        
                    elif idx ==2:
                        zip_folder(dir_model_name,templatefolder+'/'+'2_Model')
                        
                elif int(r) ==2:
                    dir_model_name=templatefolder+'/'+file_baseModel_2
                    np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_2),TrSEQLENGTH)
                    shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')
                    if idx ==0:
                        zip_folder(dir_model_name,templatefolder+'/'+'0_Model')
                    elif idx ==1:
                        zip_folder(dir_model_name,templatefolder+'/'+'1_Model')
                    elif idx ==2:
                        zip_folder(dir_model_name,templatefolder+'/'+'2_Model')
                elif int(r) >2 and int(r)!= rankA:
                    file_baseModel_Ens='{}_Model_Ens'.format(str(idx))
                    os.mkdir('%s/%s' % (templatefolder, file_baseModel_Ens))
                    dir_model_name=templatefolder+'/'+file_baseModel_Ens
                    np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_Ens),TrSEQLENGTH)
                    shutil.copy(templatefolder+'/{}_MLmodel.pkl'.format(top3Moldes_name[idx]),dir_model_name)
                    shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')
                    shutil.copy(templatefolder+'/featuresSorted_Ens.npy',dir_model_name)
                    if idx ==0:
                        zip_folder(dir_model_name,templatefolder+'/'+'0_Model') 
                    elif idx ==1:
                        zip_folder(dir_model_name,templatefolder+'/'+'1_Model') 
                    elif idx ==2:
                        zip_folder(dir_model_name,templatefolder+'/'+'2_Model') 
        else: 
            if int(r) ==0:
                dir_model_name=templatefolder+'/'+file_baseModel_0
                np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_0),TrSEQLENGTH)
                shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')
                if idx ==0:
                    zip_folder(dir_model_name,templatefolder+'/'+'0_Model')
                elif idx ==1:
                    zip_folder(dir_model_name,templatefolder+'/'+'1_Model')
                elif idx ==2:
                    zip_folder(dir_model_name,templatefolder+'/'+'2_Model')
            elif int(r) ==1:
                dir_model_name=templatefolder+'/'+file_baseModel_1 
                np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_1),TrSEQLENGTH)
                shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')
                if idx ==0:
                    zip_folder(dir_model_name,templatefolder+'/'+'0_Model')
                elif idx ==1:
                    zip_folder(dir_model_name,templatefolder+'/'+'1_Model')
                elif idx ==2:
                    zip_folder(dir_model_name,templatefolder+'/'+'2_Model')
            elif int(r) ==2:
                dir_model_name=templatefolder+'/'+file_baseModel_2
                np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_2),TrSEQLENGTH)
                shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')
                if idx ==0:
                    zip_folder(dir_model_name,templatefolder+'/'+'0_Model')
                elif idx ==1:
                    zip_folder(dir_model_name,templatefolder+'/'+'1_Model')
                elif idx ==2:
                    zip_folder(dir_model_name,templatefolder+'/'+'2_Model')
            elif int(r) >2:
                file_baseModel_Ens='{}_Model_Ens'.format(str(idx))
                os.mkdir('%s/%s' % (templatefolder, file_baseModel_Ens))
                dir_model_name=templatefolder+'/'+file_baseModel_Ens
                np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_Ens),TrSEQLENGTH)
                shutil.copy(templatefolder+'/{}_MLmodel.pkl'.format(top3Moldes_name[idx]),dir_model_name)
                shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')
                shutil.copy(templatefolder+'/featuresSorted_Ens.npy',dir_model_name)
                if idx ==0:
                    zip_folder(dir_model_name,templatefolder+'/'+'0_Model') 
                elif idx ==1:
                    zip_folder(dir_model_name,templatefolder+'/'+'1_Model') 
                elif idx ==2:
                    zip_folder(dir_model_name,templatefolder+'/'+'2_Model') 
                
    
    if args.predictfile:
        outF='results'
        os.mkdir('%s/%s' % (templatefolder, outF))
        with open(args.predictfile, "r") as handle:
            records = list(parse(handle, "fasta"))
        resultDF = pd.DataFrame()
        resultDF['seqs'] = predictfasta_dict['###']
        if args.inputType == 'fasta': 
            seqN=['_'.join(v.split('_')[0:-1]) for v in predict_seqname]
            CleavageP=[v.split('_')[-1] for v in predict_seqname]
            resultDF.insert(0, 'sequence_id', seqN)
            resultDF.insert(1, 'position', CleavageP)
        
        Count=Counter(resultDF['sequence_id'].values)
        for i,sid in enumerate(list(Count.keys())):
            thirdFN=str(i)+'_'+sid
            os.mkdir('%s/%s/%s' % (templatefolder,outF,thirdFN))
            with open(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'.fasta','w') as f_tmp:
                f_tmp.write('>'+str(records[i].id)+'\n')
                f_tmp.write(str(records[i].seq)+'\n')

        if 'AODE' in MLs:  
            
            if top3_models_rank[0]==rankA:
                aodemodelfile=templatefolder+'/'+file_baseModel_aode+'/'+AodeModelNa
                AODE_predlabels,AODE_predscores=AODE_predict(args,aodemodelfile,predata.loc[:, mrmr[:Num_features_baseModel_aode]])

                resultDF['prediction'] = AODE_predlabels
                resultDF['pro'] = AODE_predscores
                for i,sid in enumerate(list(Count.keys())):
                    thirdFN=str(i)+'_'+sid
                    result_tmp=resultDF.loc[resultDF['sequence_id']==str(sid)]
                    rANK=[str(i+1) for i in range(len(result_tmp['sequence_id'].values))]
                    result_tmp.insert(0, 'rank', rANK)
                    result_tmp.sort_values(by="pro" , inplace=True, ascending=False)
                    result_tmp.to_csv('{}/{}/{}/{}_result.csv'.format(templatefolder, outF,thirdFN,thirdFN), index=False)
                    # plot
                    if args.PLOT=='Yes':
                        visual_plot(templatefolder,args.protease,outF,thirdFN,result_tmp,sid)
                resultDF.insert(0, 'protease', [args.protease] * len(predictfasta_dict['###']))
                resultDF.to_csv(args.outputpath + '/results.csv', index=False)
                
            else: 
                if top3_models_rank[0]==0:
                    predata=predata.loc[:, baseModel_0_FETURES]
                    loadedModel=templatefolder+'/'+file_baseModel_0+'/'+top3Moldes_name[0]+'_MLmodel'
                elif top3_models_rank[0]==1:
                    predata=predata.loc[:, baseModel_1_FETURES]
                    loadedModel=templatefolder+'/'+file_baseModel_1+'/'+top3Moldes_name[0]+'_MLmodel'
                elif top3_models_rank[0]==2:
                    predata=predata.loc[:, baseModel_2_FETURES]
                    loadedModel=templatefolder+'/'+file_baseModel_2+'/'+top3Moldes_name[0]+'_MLmodel'
                elif top3_models_rank[0]>2:
                    predata=predata.loc[:, Ensemble_FEATURES]
                    loadedModel=templatefolder+'/'+top3Moldes_name[0]+'_MLmodel'
                    
                clf=load_model(loadedModel)
                prediction_predata=predict_model(clf,data=predata,raw_score=True)

                
                resultDF['prediction'] = list(prediction_predata['Label'].values)
                if 'Score_1' in prediction_predata.columns:
                    resultDF['pro'] = list(prediction_predata['Score_1'].values)
                else:
                    resultDF['pro'] = list(prediction_predata['Label'].values)
                    
                for i,sid in enumerate(list(Count.keys())):
                    thirdFN=str(i)+'_'+sid
                    result_tmp=resultDF.loc[resultDF['sequence_id']==str(sid)]
                    rANK=[str(i+1) for i in range(len(result_tmp['sequence_id'].values))]
                    result_tmp.insert(0, 'rank', rANK)
                    result_tmp.sort_values(by="pro" , inplace=True, ascending=False)
                    result_tmp.to_csv('{}/{}/{}/{}_result.csv'.format(templatefolder, outF,thirdFN,thirdFN), index=False)
                    # plot
                    if args.PLOT=='Yes':
                        visual_plot(templatefolder,args.protease,outF,thirdFN,result_tmp,sid)
                resultDF.insert(0, 'protease', [args.protease] * len(predictfasta_dict['###']))
                resultDF.to_csv(args.outputpath + '/results.csv', index=False)
        else:
            if top3_models_rank[0]==0:
                predata=predata.loc[:, baseModel_0_FETURES]
                loadedModel=templatefolder+'/'+file_baseModel_0+'/'+top3Moldes_name[0]+'_MLmodel'
            elif top3_models_rank[0]==1:
                predata=predata.loc[:, baseModel_1_FETURES]
                loadedModel=templatefolder+'/'+file_baseModel_1+'/'+top3Moldes_name[0]+'_MLmodel'
            elif top3_models_rank[0]==2:
                predata=predata.loc[:, baseModel_2_FETURES]
                loadedModel=templatefolder+'/'+file_baseModel_2+'/'+top3Moldes_name[0]+'_MLmodel'
            elif top3_models_rank[0]>2:
                predata=predata.loc[:, Ensemble_FEATURES] 
                loadedModel=templatefolder+'/'+top3Moldes_name[0]+'_MLmodel'
            clf=load_model(loadedModel)   
            prediction_predata=predict_model(clf,data=predata,raw_score=True)
            
            
            resultDF['prediction'] = list(prediction_predata['Label'].values)
            if 'Score_1' in prediction_predata.columns:
                resultDF['pro'] = list(prediction_predata['Score_1'].values)
            else:
                resultDF['pro'] = list(prediction_predata['Label'].values)
            for i,sid in enumerate(list(Count.keys())):
                thirdFN=str(i)+'_'+sid 
                result_tmp=resultDF.loc[resultDF['sequence_id']==str(sid)]
                rANK=[str(i+1) for i in range(len(result_tmp['sequence_id'].values))]
                result_tmp.insert(0, 'rank', rANK)
                result_tmp.sort_values(by="pro" , inplace=True, ascending=False)
                result_tmp.to_csv('{}/{}/{}/{}_result.csv'.format(templatefolder, outF,thirdFN,thirdFN), index=False)
                # plot
                if args.PLOT=='Yes':
                    visual_plot(templatefolder,args.protease,outF,thirdFN,result_tmp,sid)
            resultDF.insert(0, 'protease', [args.protease] * len(predictfasta_dict['###']))
            resultDF.to_csv(args.outputpath + '/results.csv', index=False)


def train_models_with_ifs(args,selectedModels,data,templatefolder,x_train,y_train,x_test,y_test,predata,fold_CV,codetype,MLs,allmatrixName,predictfasta_dict,predict_seqname,TrSEQLENGTH):

    # global TrSEQLENGTH
    print('sequence length',TrSEQLENGTH)
    mRMR_data = pd.concat([y_train,x_train], axis=1)
    NUM=mRMR_data.shape[1]
    mrmr_output=pymrmr.mRMR(data, 'MID',(NUM-1)) 
    mrmr_file=templatefolder+'/'+args.protease+'_selected_feature_list_MID.txt'
    mrmr_write = open(mrmr_file, 'a')
    for i in range(len(mrmr_output)):
        s = str(mrmr_output[i]).replace('[', '').replace(']', '') 
        s = s.replace("'", '').replace(',', '') + '\n'  
        mrmr_write.write(s)
    mrmr_write.close()
    read_mrmr = pd.read_csv( mrmr_file, header=None)
    mrmr = [list(v)[0] for v in list(read_mrmr.values)]
    #IFS
    for i in range(1,NUM):
        if selectedModels:
            x_train_new = pd.concat([x_train.loc[:, mrmr[:i]], y_train], axis=1)
            x_test_new = pd.concat([x_test.loc[:, mrmr[:i]], y_test], axis=1)

            numeric_features=[]
            for col in x_train_new.columns:
                if 'KNN' in col:
                    numeric_features.append(col)


            
            exp1=setup(data=x_train_new,target='true_label',test_data=x_test_new,session_id=123,use_gpu=False,normalize=False,remove_perfect_collinearity=False,fold=fold_CV,silent=True,verbose=False,preprocess=False,numeric_features=numeric_features)

            if len(selectedModels)==2:
                baseModel_0 = create_model(selectedModels[0])
                result_baseModel_0 = pull()
                result_baseModel_0 = result_baseModel_0[fold_CV:fold_CV+1]
                result_baseModel_0.insert(0, 'num', i)
                result_baseModel_0.to_csv("{}/10CV_IFS_combined_results_train_matrix_MID_{}_new.txt".format(templatefolder,selectedModels[0]), mode='a', header=False)
                baseModel_1 = create_model(selectedModels[1])
                result_baseModel_1 = pull()
                result_baseModel_1 = result_baseModel_1[fold_CV:fold_CV+1]
                result_baseModel_1.insert(0, 'num', i)
                result_baseModel_1.to_csv("{}/10CV_IFS_combined_results_train_matrix_MID_{}_new.txt".format(templatefolder,selectedModels[1]), mode='a', header=False)
            if len(selectedModels)==1:
                baseModel_0 = create_model(selectedModels[0])
                result_baseModel_0 = pull()
                result_baseModel_0 = result_baseModel_0[fold_CV:fold_CV+1]
                result_baseModel_0.insert(0, 'num', i)
                result_baseModel_0.to_csv("{}/10CV_IFS_combined_results_train_matrix_MID_{}_new.txt".format(templatefolder,selectedModels[0]), mode='a', header=False)
        if 'AODE' in MLs: 
            if selectedModels:
                if len(selectedModels)==2:
                    rankA=2
                if len(selectedModels)==1:
                    rankA=1
            else:
                rankA=0
            IFS_result_aode=AODE_train(args,codetype,args.protease,fold_CV,x_train.loc[:, mrmr[:i]],y_train,rankA)
            IFS_result_aode.to_csv("{}/10CV_IFS_combined_results_train_matrix_MID_{}_new.txt".format(templatefolder,'AODE'), mode='a', header=False)
    # IFS end       
    if selectedModels: 
        if len(selectedModels)==2:
            IFS_num = pd.read_csv("{}/10CV_IFS_combined_results_train_matrix_MID_{}_new.txt".format(templatefolder,selectedModels[0]),header = None)
            ACC,AUC = IFS_num.iloc[:,2],IFS_num.iloc[:,3]
            top_point = list(zip(np.arange(1,NUM),ACC,AUC))
            Num_features_baseModel_0 = sorted(top_point, key = lambda kv:(kv[1], kv[2]),reverse = True)[0][0]
            
            IFS_num = pd.read_csv("{}/10CV_IFS_combined_results_train_matrix_MID_{}_new.txt".format(templatefolder,selectedModels[1]),header = None)
            ACC,AUC = IFS_num.iloc[:,2],IFS_num.iloc[:,3]
            top_point = list(zip(np.arange(1,NUM),ACC,AUC))
            Num_features_baseModel_1 = sorted(top_point, key = lambda kv:(kv[1], kv[2]),reverse = True)[0][0]
            
        if len(selectedModels)==1:
            IFS_num = pd.read_csv("{}/10CV_IFS_combined_results_train_matrix_MID_{}_new.txt".format(templatefolder,selectedModels[0]),header = None)
            ACC,AUC = IFS_num.iloc[:,2],IFS_num.iloc[:,3]
            top_point = list(zip(np.arange(1,NUM),ACC,AUC))
            Num_features_baseModel_0 = sorted(top_point, key = lambda kv:(kv[1], kv[2]),reverse = True)[0][0]

    if 'AODE' in MLs:   
        IFS_num = pd.read_csv("{}/10CV_IFS_combined_results_train_matrix_MID_{}_new.txt".format(templatefolder,'AODE'),header = None)
        ACC,AUC = IFS_num.iloc[:,3],IFS_num.iloc[:,4] 
        top_point = list(zip(np.arange(1,NUM),ACC,AUC))
        Num_features_baseModel_aode = sorted(top_point, key = lambda kv:(kv[1], kv[2]),reverse = True)[0][0]

    if selectedModels:
        if len(selectedModels)==2:
            baseModel_0_FETURES = mrmr[:Num_features_baseModel_0] 
            baseModel_1_FETURES = mrmr[:Num_features_baseModel_1]
            file_baseModel_0='0_baseModel_results'
            os.mkdir('%s/%s' % (templatefolder, file_baseModel_0))
            np.save('{}/{}/featuresSorted.npy'.format(templatefolder,file_baseModel_0),baseModel_0_FETURES)
            file_baseModel_1='1_baseModel_results'
            os.mkdir('%s/%s' % (templatefolder, file_baseModel_1))
            np.save('{}/{}/featuresSorted.npy'.format(templatefolder,file_baseModel_1),baseModel_1_FETURES)
        elif len(selectedModels)==1:
            baseModel_0_FETURES = mrmr[:Num_features_baseModel_0]
            file_baseModel_0='0_baseModel_results'
            os.mkdir('%s/%s' % (templatefolder, file_baseModel_0))
            np.save('{}/{}/featuresSorted.npy'.format(templatefolder,file_baseModel_0),baseModel_0_FETURES)

    
    train_X = x_train
    train_y = y_train
    test_X = x_test
    test_y = y_test
    # get models with best features
    if selectedModels:
        if len(selectedModels)==2:         
            x_train_new = pd.concat([x_train.loc[:, baseModel_0_FETURES], y_train], axis=1)
            x_test_new = pd.concat([x_test.loc[:, baseModel_0_FETURES], y_test], axis=1)
            numeric_features=[]
            for col in x_train_new.columns:
                if 'KNN' in col:
                    numeric_features.append(col)
                    
            exp11=setup(data=x_train_new,target='true_label',test_data=x_test_new,session_id=123,use_gpu=False,normalize=False,remove_perfect_collinearity=False,fold=fold_CV,silent=True,verbose=False,preprocess=False,numeric_features=numeric_features)
            baseModel_0_with_OptFEATURES = create_model(selectedModels[0])
            CVresult_bM_0_wOptF=pull()
            CVresult_bM_0_wOptF =CVresult_bM_0_wOptF[fold_CV:fold_CV+1]
            CVresult_bM_0_wOptF.insert(0,'Model',selectedModels[0])
            CVresult_bM_0_wOptF.insert(0,'model_rank',0)
            CVresult_bM_0_wOptF=CVresult_bM_0_wOptF.drop('Kappa',1)
            save_model(baseModel_0_with_OptFEATURES,templatefolder+'/'+file_baseModel_0+'/'+selectedModels[0]+'_MLmodel')
            
            x_train_new1 = pd.concat([x_train.loc[:, baseModel_1_FETURES], y_train], axis=1)
            x_test_new1 = pd.concat([x_test.loc[:, baseModel_1_FETURES], y_test], axis=1)
            numeric_features=[]
            for col in x_train_new1.columns:
                if 'KNN' in col:
                    numeric_features.append(col)
            exp12=setup(data=x_train_new1,target='true_label',test_data=x_test_new1,session_id=123,use_gpu=False,normalize=False,remove_perfect_collinearity=False,fold=fold_CV,silent=True,verbose=False,preprocess=False,numeric_features=numeric_features)

            baseModel_1_with_OptFEATURES = create_model(selectedModels[1])
            CVresult_bM_1_wOptF=pull()
            CVresult_bM_1_wOptF =CVresult_bM_1_wOptF[fold_CV:fold_CV+1]
            CVresult_bM_1_wOptF.insert(0,'Model',selectedModels[1])
            CVresult_bM_1_wOptF.insert(0,'model_rank',1)
            CVresult_bM_1_wOptF=CVresult_bM_1_wOptF.drop('Kappa',1)
            save_model(baseModel_1_with_OptFEATURES,templatefolder+'/'+file_baseModel_1+'/'+selectedModels[1]+'_MLmodel')
            
        if len(selectedModels)==1:
            x_train_new = pd.concat([x_train.loc[:, baseModel_0_FETURES], y_train], axis=1)
            x_test_new = pd.concat([x_test.loc[:, baseModel_0_FETURES], y_test], axis=1)
            numeric_features=[]
            for col in x_train_new.columns:
                if 'KNN' in col:
                    numeric_features.append(col)
            exp11=setup(data=x_train_new,target='true_label',test_data=x_test_new,session_id=123,use_gpu=False,normalize=False,remove_perfect_collinearity=False,fold=fold_CV,silent=True,verbose=False,preprocess=False,numeric_features=numeric_features)

            baseModel_0_with_OptFEATURES = create_model(selectedModels[0])
            CVresult_bM_0_wOptF=pull()
            CVresult_bM_0_wOptF =CVresult_bM_0_wOptF[fold_CV:fold_CV+1]
            CVresult_bM_0_wOptF.insert(0,'Model',selectedModels[0])
            CVresult_bM_0_wOptF.insert(0,'model_rank',0)
            CVresult_bM_0_wOptF=CVresult_bM_0_wOptF.drop('Kappa',1)
            save_model(baseModel_0_with_OptFEATURES,templatefolder+'/'+file_baseModel_0+'/'+selectedModels[0]+'_MLmodel')

    if 'AODE' in MLs:
        baseModel_aode_FETURES = mrmr[:Num_features_baseModel_aode]
        file_baseModel_aode='{}_baseModel_results'.format(str(rankA))
        os.mkdir('%s/%s' % (templatefolder, file_baseModel_aode))
        np.save('{}/{}/features_sorted.npy'.format(templatefolder,file_baseModel_aode),baseModel_aode_FETURES)
        np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_aode),TrSEQLENGTH)
        aodeCVpred,aode_test_pred,aode_test_predLabel,AodeModelNa=AODE_trainAndPred(args,codetype,args.protease,fold_CV,x_train.loc[:, baseModel_aode_FETURES],y_train,x_test.loc[:, baseModel_aode_FETURES],y_test,rankA)
        shutil.move(templatefolder+'/'+AodeModelNa,templatefolder+'/'+file_baseModel_aode)
        shutil.copy(allmatrixName,templatefolder+'/'+file_baseModel_aode+'/'+'allmatrix.json')
        
    if selectedModels:   
        if len(selectedModels)==2 and 'AODE' in MLs:
            alltrainedModel_pred_metric=pd.concat([CVresult_bM_0_wOptF,CVresult_bM_1_wOptF,aodeCVpred],ignore_index=True)
            alltrainedModel_pred_metric['indx']=selectedModels+['AODE']
            alltrainedModel_pred_metric.set_index('indx',inplace=True)    
        if len(selectedModels)==1 and 'AODE' in MLs:
            alltrainedModel_pred_metric=pd.concat([CVresult_bM_0_wOptF,aodeCVpred],ignore_index=True)
            alltrainedModel_pred_metric['indx']=selectedModels+['AODE']
            alltrainedModel_pred_metric.set_index('indx',inplace=True)
        if len(selectedModels)==2 and 'AODE' not in MLs:
            alltrainedModel_pred_metric=pd.concat([CVresult_bM_0_wOptF,CVresult_bM_1_wOptF],ignore_index=True)
            alltrainedModel_pred_metric['indx']=selectedModels
            alltrainedModel_pred_metric.set_index('indx',inplace=True)    
        if len(selectedModels)==1 and 'AODE' not in MLs:
            alltrainedModel_pred_metric=pd.concat([CVresult_bM_0_wOptF],ignore_index=True)
            alltrainedModel_pred_metric['indx']=selectedModels
            alltrainedModel_pred_metric.set_index('indx',inplace=True) 

        alltrainedModel_pred_metric.sort_values(by="Accuracy" , inplace=True, ascending=False) # AUC,Accuracy

        alltrainedModel_pred_metric.to_csv(templatefolder+'/allModels_CVresults.txt',index=False)   
        topMoldes_name=alltrainedModel_pred_metric['Model'].values[0:3]
        
        if len(selectedModels)==2:
            All_trained_models=[baseModel_0_with_OptFEATURES,baseModel_1_with_OptFEATURES]
        if len(selectedModels)==1:   
            All_trained_models=[baseModel_0_with_OptFEATURES]
            

        top3_models_rank=alltrainedModel_pred_metric['model_rank'].values[0:3]
 
        top3Models=[]
        for i in top3_models_rank:
            if 'AODE' in MLs:
                if int(i)==rankA:
                    top3Models.append('AODE')
                    # pass
                else:
                    top3Models.append(All_trained_models[i]) 
            else:
                top3Models.append(All_trained_models[i]) 
        if args.testfile:
            
            labelTag=[]
            fprList=[]
            tprList=[]
            aucscore=[]
            
            checkpoint=0
            checkpoint1=0
            if 'AODE' not in MLs:
                rankA=''
            for i,m in enumerate(list(top3_models_rank)):

                if 'AODE' in MLs:
                    if int(m)==rankA:
                        fpr,tpr, thresholds=roc_curve(y_test.values,aode_test_predLabel) 
                        rocTag='ROC of AODE'
                        auroc=auc(fpr,tpr)
                        fprList.append(fpr)
                        tprList.append(tpr)
                        aucscore.append(auroc)
                        labelTag.append(rocTag)
                        if checkpoint==0:
                            top3pred_metrics=aode_test_pred
                            checkpoint=1
                        else:
                            top3pred_metrics=top3pred_metrics.append(aode_test_pred)
                
                if int(m)==rankA and 'AODE' in MLs:
                    continue
                else:
                    if len(selectedModels)==2:
                        if int(m)==0:
                            loadedModel=templatefolder+'/'+file_baseModel_0+'/'+topMoldes_name[0]+'_MLmodel'
                            clf_=load_model(loadedModel)
                            set_config('X_test', test_X.loc[:, baseModel_0_FETURES])
                        elif int(m)==1:
                            loadedModel=templatefolder+'/'+file_baseModel_1+'/'+topMoldes_name[0]+'_MLmodel'
                            clf_=load_model(loadedModel)
                            set_config('X_test', test_X.loc[:,baseModel_1_FETURES])
                    elif len(selectedModels)==1:
                        if int(m)==0:
                            loadedModel=templatefolder+'/'+file_baseModel_0+'/'+topMoldes_name[0]+'_MLmodel'
                            clf_=load_model(loadedModel)
                            set_config('X_test', test_X.loc[:, baseModel_0_FETURES])
                    
                    pred=predict_model(clf_,raw_score=True)
                    pred.to_csv(templatefolder+'/testdata_pre.txt')
                    pred_metrics=pull()
                    pred_metrics.insert(1,'Model_name',topMoldes_name[0])

                    pred_metrics=pred_metrics.drop(['Kappa'],1)

                    if checkpoint==0:
                        
                        top3pred_metrics=pred_metrics
                        checkpoint=1
                    else:
                        top3pred_metrics=top3pred_metrics.append(pred_metrics)

                    if 'Score_1' in pred.columns:
                        pre_s=pred['Score_1']
                    else:
                        pre_s=pred['Label']
                    fpr,tpr, thresholds=roc_curve(pred['true_label'],pre_s) # 

                    rocTag='ROC of '+topMoldes_name[0]
                    auroc=auc(fpr,tpr)
                    fprList.append(fpr)
                    tprList.append(tpr)
                    aucscore.append(auroc)
                    labelTag.append(rocTag)
            
            
            # plot ROC
            plotAuc(fprList, tprList, aucscore, labelTag, False,templatefolder)
        # plot SHAP
        if selectedModels:
            if args.SHAP=='Yes': 
                if len(selectedModels)==2:
                    if int(top3_models_rank[0])==0:
                       
                        test_SHAP=test_X.loc[:, mrmr[:Num_features_baseModel_0]]
                        train_SHAP=train_X.loc[:, mrmr[:Num_features_baseModel_0]]
                    elif int(top3_models_rank[0])==1:
                        test_SHAP=test_X.loc[:, mrmr[:Num_features_baseModel_1]]
                        train_SHAP=train_X.loc[:, mrmr[:Num_features_baseModel_1]]
                else:

                    test_SHAP=test_X.loc[:, mrmr[:Num_features_baseModel_0]]
                    train_SHAP=train_X.loc[:, mrmr[:Num_features_baseModel_0]]

                SHAP_plot(args,templatefolder,topMoldes_name,top3Models, train_SHAP,test_SHAP) 
                
        if 'AODE' not in topMoldes_name and 'AODE' in MLs:
            if codetype=='all':
                os.remove('%s/test.arff' % (templatefolder))
                os.remove('%s/train.arff' % (templatefolder))
                os.remove('%s/testresult.txt' % (templatefolder))
                os.remove('%s/trainresult.txt' % (templatefolder))
                os.remove('%s/*_AODE_model' % (templatefolder))

        if args.testfile:
            finaltop3pred_metrics=top3pred_metrics.iloc[0:3,1:]
            finaltop3pred_metrics.to_csv(templatefolder+'/TestResults.txt',index=False)
        for idx,r in enumerate(top3_models_rank):
            if 'AODE' in MLs:
                if r==rankA :
                    if codetype=='all':
                        os.remove('%s/test.arff' % (templatefolder))
                        os.remove('%s/train.arff' % (templatefolder))
                        os.remove('%s/testresult.txt' % (templatefolder))
                        os.remove('%s/trainresult.txt' % (templatefolder))
                        if idx ==0:
                            zip_folder(templatefolder+'/'+file_baseModel_aode, templatefolder+'/'+'0_Model')
                        elif idx ==1:
                            zip_folder(templatefolder+'/'+file_baseModel_aode, templatefolder+'/'+'1_Model')
                        elif idx ==2:
                            zip_folder(templatefolder+'/'+file_baseModel_aode, templatefolder+'/'+'2_Model')
                else:
                    if len(selectedModels)==2:
                        if int(r) ==0:
                            dir_model_name=templatefolder+'/'+file_baseModel_0
                            np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_0),TrSEQLENGTH)
                            shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')

                            if idx ==0:
                                zip_folder(dir_model_name,templatefolder+'/'+'0_Model')
                            elif idx ==1:
                                zip_folder(dir_model_name,templatefolder+'/'+'1_Model')
                            elif idx ==2:  
                                zip_folder(dir_model_name,templatefolder+'/'+'2_Model') 
                        elif int(r) ==1:
                            dir_model_name=templatefolder+'/'+file_baseModel_1 

                            np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_1),TrSEQLENGTH)
                            shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')
                            if idx ==0:
                                zip_folder(dir_model_name,templatefolder+'/'+'0_Model')
                            elif idx ==1:
                                zip_folder(dir_model_name,templatefolder+'/'+'1_Model')
                            elif idx ==2:  
                                zip_folder(dir_model_name,templatefolder+'/'+'2_Model') 

                    if len(selectedModels)==1:
                        if int(r) ==0:
                            dir_model_name=templatefolder+'/'+file_baseModel_0
                            np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_0),TrSEQLENGTH)
                            shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')
                            if idx ==0:
                                zip_folder(dir_model_name,templatefolder+'/'+'0_Model')
                            elif idx ==1:
                                zip_folder(dir_model_name,templatefolder+'/'+'1_Model')

            else:
                if int(r) ==0:
                    dir_model_name=templatefolder+'/'+file_baseModel_0
                    np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_0),TrSEQLENGTH)
                    shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')

                    if idx ==0:
                        zip_folder(dir_model_name,templatefolder+'/'+'0_Model')
                    elif idx ==1:
                        zip_folder(dir_model_name,templatefolder+'/'+'1_Model')
                elif int(r) ==1:
                    dir_model_name=templatefolder+'/'+file_baseModel_1 
                    np.save('{}/{}/TrSEQLENGTH.npy'.format(templatefolder,file_baseModel_1),TrSEQLENGTH)
                    shutil.copy(allmatrixName,dir_model_name+'/'+'allmatrix.json')

                    if idx ==0:
                        zip_folder(dir_model_name,templatefolder+'/'+'0_Model')
                    elif idx ==1:
                        zip_folder(dir_model_name,templatefolder+'/'+'1_Model')

    else:
        labelTag=[]
        fprList=[]
        tprList=[]
        aucscore=[]  
        fpr,tpr, thresholds=roc_curve(y_test.values,aode_test_predLabel) # 
        rocTag='ROC of AODE'
        auroc=auc(fpr,tpr)
        fprList.append(fpr)
        tprList.append(tpr)
        aucscore.append(auroc)
        labelTag.append(rocTag)  
        # plot ROC
        plotAuc(fprList, tprList, aucscore, labelTag, False,templatefolder)
        aodeCVpred.to_csv(templatefolder+'/allModels_CVresults.txt',index=False) 
        aode_test_pred.to_csv(templatefolder+'/TestResults.txt',index=False)
        if codetype=='all':
            os.remove('%s/test.arff' % (templatefolder))
            os.remove('%s/train.arff' % (templatefolder))
            os.remove('%s/testresult.txt' % (templatefolder)) 
            os.remove('%s/trainresult.txt' % (templatefolder)) 
        zip_folder(templatefolder+'/'+file_baseModel_aode,templatefolder+'/'+'0_Model') 
              
    if args.predictfile:
        outF='results'
        os.mkdir('%s/%s' % (templatefolder, outF))
        with open(args.predictfile, "r") as handle:
            records = list(parse(handle, "fasta"))

        resultDF = pd.DataFrame()
        resultDF['seqs'] = predictfasta_dict['###']
        
        if args.inputType == 'fasta':
            seqN=['_'.join(v.split('_')[0:-1]) for v in predict_seqname]
            CleavageP=[v.split('_')[-1] for v in predict_seqname]
            resultDF.insert(0, 'sequence_id', seqN)
            resultDF.insert(1, 'position', CleavageP)
            
        Count=Counter(resultDF['sequence_id'].values)
        for i,sid in enumerate(list(Count.keys())):
            thirdFN=str(i)+'_'+sid
            os.mkdir('%s/%s/%s' % (templatefolder,outF,thirdFN))
            with open(templatefolder+'/'+outF+'/'+thirdFN+'/'+thirdFN+'.fasta','w') as f_tmp:
                f_tmp.write('>'+str(records[i].id)+'\n')
                f_tmp.write(str(records[i].seq)+'\n')
        
           
             


        if selectedModels:
            if 'AODE' in MLs:   
                if top3_models_rank[0]==rankA:
                    aodemodelfile=templatefolder+'/'+file_baseModel_aode+'/'+AodeModelNa
                    predata=predata.loc[:, baseModel_aode_FETURES]
                    AODE_predlabels,AODE_predscores=AODE_predict(args,aodemodelfile,predata)

                    resultDF['prediction'] = AODE_predlabels
                    resultDF['pro'] = AODE_predscores
                    for i,sid in enumerate(list(Count.keys())):
                        thirdFN=str(i)+'_'+sid
                        
                        result_tmp=resultDF.loc[resultDF['sequence_id']==str(sid)]
                        rANK=[str(i+1) for i in range(len(result_tmp['sequence_id'].values))]
                        result_tmp.insert(0, 'rank', rANK)
                        result_tmp.sort_values(by="pro" , inplace=True, ascending=False)
                        result_tmp.to_csv('{}/{}/{}/{}_result.csv'.format(templatefolder, outF,thirdFN,thirdFN), index=False)
                        # plot
                        if args.PLOT=='Yes':
                            visual_plot(templatefolder,args.protease,outF,thirdFN,result_tmp,sid) 
                    resultDF.insert(0, 'protease', [args.protease] * len(predictfasta_dict['###']))
                    resultDF.to_csv(args.outputpath + '/results.csv', index=False)
                else: 
                    if top3_models_rank[0]==0:
                        predata=predata.loc[:, mrmr[:Num_features_baseModel_0]]
                        loadedModel=templatefolder+'/'+file_baseModel_0+'/'+topMoldes_name[0]+'_MLmodel'
                    elif top3_models_rank[0]==1:
                        predata=predata.loc[:, mrmr[:Num_features_baseModel_1]]
                        loadedModel=templatefolder+'/'+file_baseModel_1+'/'+topMoldes_name[0]+'_MLmodel'
                        
                    clf=load_model(loadedModel)
                    prediction_predata=predict_model(clf,data=predata,raw_score=True)


                    resultDF['prediction'] = list(prediction_predata['Label'].values)
                    if 'Score_1' in prediction_predata.columns:
                        resultDF['pro'] = list(prediction_predata['Score_1'].values)
                    else:
                        resultDF['pro'] = list(prediction_predata['Label'].values)
                        
                    for i,sid in enumerate(list(Count.keys())):
                        thirdFN=str(i)+'_'+sid
                        result_tmp=resultDF.loc[resultDF['sequence_id']==str(sid)]
                        rANK=[str(i+1) for i in range(len(result_tmp['sequence_id'].values))]
                        result_tmp.insert(0, 'rank', rANK)
                        result_tmp.sort_values(by="pro" , inplace=True, ascending=False)
                        result_tmp.to_csv('{}/{}/{}/{}_result.csv'.format(templatefolder, outF,thirdFN,thirdFN), index=False)
                        # plot
                        if args.PLOT=='Yes':
                            visual_plot(templatefolder,args.protease,outF,thirdFN,result_tmp,sid)
                    resultDF.insert(0, 'protease', [args.protease] * len(predictfasta_dict['###']))
                    resultDF.to_csv(args.outputpath + '/results.csv', index=False)
  
            else:
                if top3_models_rank[0]==0:
                    predata=predata.loc[:, mrmr[:Num_features_baseModel_0]]
                    loadedModel=templatefolder+'/'+file_baseModel_0+'/'+topMoldes_name[0]+'_MLmodel'
                elif top3_models_rank[0]==1:
                    predata=predata.loc[:, mrmr[:Num_features_baseModel_1]]
                    loadedModel=templatefolder+'/'+file_baseModel_1+'/'+topMoldes_name[0]+'_MLmodel'
                clf=load_model(loadedModel)
                prediction_predata=predict_model(clf,data=predata,raw_score=True)


                resultDF['prediction'] = list(prediction_predata['Label'].values)
                if 'Score_1' in prediction_predata.columns:
                    resultDF['pro'] = list(prediction_predata['Score_1'].values)
                else:
                    resultDF['pro'] = list(prediction_predata['Label'].values)
                for i,sid in enumerate(list(Count.keys())):
                    thirdFN=str(i)+'_'+sid
                    
                    result_tmp=resultDF.loc[resultDF['sequence_id']==str(sid)]
                    rANK=[str(i+1) for i in range(len(result_tmp['sequence_id'].values))]
                    result_tmp.insert(0, 'rank', rANK)
                    result_tmp.sort_values(by="pro" , inplace=True, ascending=False)
                    result_tmp.to_csv('{}/{}/{}/{}_result.csv'.format(templatefolder, outF,thirdFN,thirdFN), index=False)
                    # plot
                    if args.PLOT=='Yes':
                        visual_plot(templatefolder,args.protease,outF,thirdFN,result_tmp,sid)
                resultDF.insert(0, 'protease', [args.protease] * len(predictfasta_dict['###']))
                resultDF.to_csv(args.outputpath + '/results.csv', index=False)
        else:
            aodemodelfile=templatefolder+'/'+file_baseModel_aode+'/'+AodeModelNa
            predata=predata.loc[:, baseModel_aode_FETURES]
            AODE_predlabels,AODE_predscores=AODE_predict(args,aodemodelfile,predata)

            resultDF['prediction'] = AODE_predlabels
            resultDF['pro'] = AODE_predscores
            for i,sid in enumerate(list(Count.keys())):
                thirdFN=str(i)+'_'+sid
                
                result_tmp=resultDF.loc[resultDF['sequence_id']==str(sid)]
                rANK=[str(i+1) for i in range(len(result_tmp['sequence_id'].values))]
                result_tmp.insert(0, 'rank', rANK)
                result_tmp.sort_values(by="pro" , inplace=True, ascending=False)
                result_tmp.to_csv('{}/{}/{}/{}_result.csv'.format(templatefolder, outF,thirdFN,thirdFN), index=False)
                # plot
                if args.PLOT=='Yes':
                    visual_plot(templatefolder,args.protease,outF,thirdFN,result_tmp,sid)
            resultDF.insert(0, 'protease', [args.protease] * len(predictfasta_dict['###']))
            resultDF.to_csv(args.outputpath + '/results.csv', index=False) 
            # os.system('rm -r %s/predict.arff' % (templatefolder))
            os.remove('%s/predict.arff' % (templatefolder))
