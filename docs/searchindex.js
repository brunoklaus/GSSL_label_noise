Search.setIndex({docnames:["_modules/experiment","_modules/experiment.hooks","_modules/experiment.specification","_modules/gssl","_modules/gssl.classifiers","_modules/gssl.classifiers.nn","_modules/gssl.filters","_modules/gssl.graph","_modules/input","_modules/input.dataset","_modules/input.noise","_modules/log","_modules/modules","_modules/output","_modules/settings","index","zreferences"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":2,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.todo":2,sphinx:56},filenames:["_modules/experiment.rst","_modules/experiment.hooks.rst","_modules/experiment.specification.rst","_modules/gssl.rst","_modules/gssl.classifiers.rst","_modules/gssl.classifiers.nn.rst","_modules/gssl.filters.rst","_modules/gssl.graph.rst","_modules/input.rst","_modules/input.dataset.rst","_modules/input.noise.rst","_modules/log.rst","_modules/modules.rst","_modules/output.rst","_modules/settings.rst","index.rst","zreferences.rst"],objects:{"":{experiment:[0,0,0,"-"],gssl:[3,0,0,"-"],input:[8,0,0,"-"],log:[11,0,0,"-"],output:[13,0,0,"-"],settings:[14,0,0,"-"]},"experiment.experiments":{Experiment:[0,1,1,""],keys_multiplex:[0,4,1,""],main:[0,4,1,""],postprocess:[0,4,1,""],run_debug_example_all:[0,4,1,""],run_debug_example_one:[0,4,1,""]},"experiment.experiments.Experiment":{W:[0,2,1,""],X:[0,2,1,""],Y:[0,2,1,""],__init__:[0,3,1,""],run:[0,3,1,""]},"experiment.hooks":{hook_skeleton:[1,0,0,"-"],ldst_filterstats_hook:[1,0,0,"-"],plot_hooks:[1,0,0,"-"],time_hook:[1,0,0,"-"]},"experiment.hooks.hook_skeleton":{CompositeHook:[1,1,1,""],GSSLHook:[1,1,1,""]},"experiment.hooks.hook_skeleton.CompositeHook":{__init__:[1,3,1,""]},"experiment.hooks.hook_skeleton.GSSLHook":{get_step_rate:[1,3,1,""]},"experiment.hooks.ldst_filterstats_hook":{LDSTFilterStatsHook:[1,1,1,""]},"experiment.hooks.ldst_filterstats_hook.LDSTFilterStatsHook":{__init__:[1,3,1,""],best_f1:[1,2,1,""]},"experiment.hooks.plot_hooks":{plotHook:[1,1,1,""],plotIterGTAMHook:[1,1,1,""],plotIterHook:[1,1,1,""]},"experiment.hooks.plot_hooks.plotHook":{__init__:[1,3,1,""],plot:[1,3,1,""]},"experiment.hooks.plot_hooks.plotIterGTAMHook":{__init__:[1,3,1,""]},"experiment.hooks.plot_hooks.plotIterHook":{__init__:[1,3,1,""],createVideo:[1,3,1,""],rmFolders:[1,3,1,""]},"experiment.hooks.time_hook":{timeHook:[1,1,1,""]},"experiment.hooks.time_hook.timeHook":{__init__:[1,3,1,""]},"experiment.selector":{Hook:[0,1,1,""],HookTimes:[0,1,1,""],select_affmat:[0,4,1,""],select_and_add_hook:[0,4,1,""],select_classifier:[0,4,1,""],select_filter:[0,4,1,""],select_input:[0,4,1,""],select_noise:[0,4,1,""]},"experiment.selector.Hook":{ALG_ITER:[0,2,1,""],ALG_RESULT:[0,2,1,""],FILTER_AFTER:[0,2,1,""],FILTER_ITER:[0,2,1,""],GTAM_F:[0,2,1,""],GTAM_Q:[0,2,1,""],GTAM_Y:[0,2,1,""],INIT_ALL:[0,2,1,""],INIT_LABELED:[0,2,1,""],LDST_STATS_HOOK:[0,2,1,""],NOISE_AFTER:[0,2,1,""],T_AFFMAT:[0,2,1,""],T_ALG:[0,2,1,""],T_FILTER:[0,2,1,""],T_NOISE:[0,2,1,""],W_FILTER_AFTER:[0,2,1,""],W_INIT_ALL:[0,2,1,""],W_INIT_LABELED:[0,2,1,""],W_NOISE_AFTER:[0,2,1,""]},"experiment.selector.HookTimes":{AFTER_AFFMAT:[0,2,1,""],AFTER_CLASSIFIER:[0,2,1,""],AFTER_FILTER:[0,2,1,""],AFTER_NOISE:[0,2,1,""],BEFORE_AFFMAT:[0,2,1,""],BEFORE_CLASSIFIER:[0,2,1,""],BEFORE_FILTER:[0,2,1,""],BEFORE_NOISE:[0,2,1,""],DURING_AFFMAT:[0,2,1,""],DURING_CLASSIFIER:[0,2,1,""],DURING_FILTER:[0,2,1,""],DURING_NOISE:[0,2,1,""]},"experiment.specification":{exp_chapelle:[2,0,0,"-"],exp_cifar:[2,0,0,"-"],exp_debug:[2,0,0,"-"],exp_filter_LDST:[2,0,0,"-"],specification_bits:[2,0,0,"-"],specification_skeleton:[2,0,0,"-"]},"experiment.specification.exp_chapelle":{ExpChapelle:[2,1,1,""],ExpChapelle_2:[2,1,1,""],ExpChapelle_3:[2,1,1,""],ExpChapelle_4:[2,1,1,""],ExpChapelle_5:[2,1,1,""]},"experiment.specification.exp_chapelle.ExpChapelle":{__init__:[2,3,1,""],affmatConfig:[2,3,1,""],algConfig:[2,3,1,""],ds:[2,2,1,""],filterConfig:[2,3,1,""],generalConfig:[2,3,1,""],get_spec_name:[2,3,1,""],inputConfig:[2,3,1,""],noiseConfig:[2,3,1,""]},"experiment.specification.exp_chapelle.ExpChapelle_2":{affmatConfig:[2,3,1,""],algConfig:[2,3,1,""],filterConfig:[2,3,1,""],generalConfig:[2,3,1,""],get_spec_name:[2,3,1,""],inputConfig:[2,3,1,""],noiseConfig:[2,3,1,""]},"experiment.specification.exp_chapelle.ExpChapelle_3":{affmatConfig:[2,3,1,""],algConfig:[2,3,1,""],filterConfig:[2,3,1,""],generalConfig:[2,3,1,""],get_spec_name:[2,3,1,""],inputConfig:[2,3,1,""],noiseConfig:[2,3,1,""]},"experiment.specification.exp_chapelle.ExpChapelle_4":{affmatConfig:[2,3,1,""],algConfig:[2,3,1,""],filterConfig:[2,3,1,""],generalConfig:[2,3,1,""],get_spec_name:[2,3,1,""],inputConfig:[2,3,1,""],noiseConfig:[2,3,1,""]},"experiment.specification.exp_chapelle.ExpChapelle_5":{affmatConfig:[2,3,1,""],algConfig:[2,3,1,""],filterConfig:[2,3,1,""],generalConfig:[2,3,1,""],get_spec_name:[2,3,1,""],inputConfig:[2,3,1,""],noiseConfig:[2,3,1,""]},"experiment.specification.exp_cifar":{ExpCIFAR:[2,1,1,""]},"experiment.specification.exp_cifar.ExpCIFAR":{affmatConfig:[2,3,1,""],algConfig:[2,3,1,""],filterConfig:[2,3,1,""],generalConfig:[2,3,1,""],get_spec_name:[2,3,1,""],inputConfig:[2,3,1,""],noiseConfig:[2,3,1,""]},"experiment.specification.exp_debug":{ExpDebug:[2,1,1,""]},"experiment.specification.exp_debug.ExpDebug":{affmatConfig:[2,3,1,""],algConfig:[2,3,1,""],filterConfig:[2,3,1,""],generalConfig:[2,3,1,""],get_spec_name:[2,3,1,""],inputConfig:[2,3,1,""],noiseConfig:[2,3,1,""]},"experiment.specification.exp_filter_LDST":{FilterLDST:[2,1,1,""],ISOLET:[2,1,1,""],MNIST:[2,1,1,""]},"experiment.specification.exp_filter_LDST.FilterLDST":{WRITE_FREQ:[2,2,1,""],affmatConfig:[2,3,1,""],algConfig:[2,3,1,""],filterConfig:[2,3,1,""],generalConfig:[2,3,1,""],get_spec_name:[2,3,1,""],inputConfig:[2,3,1,""],noiseConfig:[2,3,1,""],run:[2,3,1,""]},"experiment.specification.exp_filter_LDST.ISOLET":{WRITE_FREQ:[2,2,1,""],affmatConfig:[2,3,1,""],algConfig:[2,3,1,""],filterConfig:[2,3,1,""],generalConfig:[2,3,1,""],get_spec_name:[2,3,1,""],inputConfig:[2,3,1,""],noiseConfig:[2,3,1,""]},"experiment.specification.exp_filter_LDST.MNIST":{WRITE_FREQ:[2,2,1,""],affmatConfig:[2,3,1,""],algConfig:[2,3,1,""],filterConfig:[2,3,1,""],generalConfig:[2,3,1,""],get_spec_name:[2,3,1,""],inputConfig:[2,3,1,""],noiseConfig:[2,3,1,""]},"experiment.specification.specification_bits":{AFFMAT_CIFAR10_LOAD:[2,5,1,""],ALGORITHM_NONE:[2,5,1,""],FILTER_MR:[2,5,1,""],FILTER_NOFILTER:[2,5,1,""],NOISE_UNIFORM_DET_MODERATE:[2,5,1,""],add_key_prefix:[2,4,1,""],allPermutations:[2,4,1,""],comb:[2,4,1,""]},"experiment.specification.specification_skeleton":{EmptySpecification:[2,1,1,""],processify:[2,4,1,""],runprocessify_func:[2,4,1,""]},"experiment.specification.specification_skeleton.EmptySpecification":{DEBUG_MODE:[2,2,1,""],FORCE_GTAM_LDST_SAME_MU:[2,2,1,""],OVERWRITE:[2,2,1,""],TUNING_ITER_AS_NOISE_PCT:[2,2,1,""],WRITE_FREQ:[2,2,1,""],affmatConfig:[2,3,1,""],aggregate_csv:[2,3,1,""],algConfig:[2,3,1,""],filterConfig:[2,3,1,""],generalConfig:[2,3,1,""],get_all_configs:[2,3,1,""],get_spec_name:[2,3,1,""],inputConfig:[2,3,1,""],noiseConfig:[2,3,1,""],run:[2,3,1,""],run_all:[2,3,1,""]},"gssl.classifiers":{CLGC:[4,0,0,"-"],GFHF:[4,0,0,"-"],GTAM:[4,0,0,"-"],LGC:[4,0,0,"-"],LGC_tf:[4,0,0,"-"],LapEigLS:[4,0,0,"-"],RF:[4,0,0,"-"],SIIS:[4,0,0,"-"],classifier:[4,0,0,"-"],nn:[5,0,0,"-"]},"gssl.classifiers.CLGC":{CLGCClassifier:[4,1,1,""]},"gssl.classifiers.CLGC.CLGCClassifier":{__init__:[4,3,1,""],alpha:[4,3,1,""],fit:[4,3,1,""]},"gssl.classifiers.GFHF":{GFHF:[4,1,1,""]},"gssl.classifiers.GFHF.GFHF":{__init__:[4,3,1,""],fit:[4,3,1,""]},"gssl.classifiers.GTAM":{GTAMClassifier:[4,1,1,""]},"gssl.classifiers.GTAM.GTAMClassifier":{__init__:[4,3,1,""],fit:[4,3,1,""]},"gssl.classifiers.LGC":{LGCClassifier:[4,1,1,""]},"gssl.classifiers.LGC.LGCClassifier":{__init__:[4,3,1,""],alpha:[4,3,1,""],fit:[4,3,1,""]},"gssl.classifiers.LGC_tf":{LGC_iter_TF:[4,4,1,""],convert_sparse_matrix_to_sparse_tensor:[4,4,1,""],gather:[4,4,1,""],get_P:[4,4,1,""],get_S_fromtensor:[4,4,1,""],repeat:[4,4,1,""],row_normalize:[4,4,1,""],update_F:[4,4,1,""]},"gssl.classifiers.LapEigLS":{LapEigLS:[4,1,1,""]},"gssl.classifiers.LapEigLS.LapEigLS":{__init__:[4,3,1,""],fit:[4,3,1,""]},"gssl.classifiers.RF":{RFClassifier:[4,1,1,""]},"gssl.classifiers.RF.RFClassifier":{__init__:[4,3,1,""],fit:[4,3,1,""]},"gssl.classifiers.SIIS":{SIISClassifier:[4,1,1,""]},"gssl.classifiers.SIIS.SIISClassifier":{__init__:[4,3,1,""],edge_mat:[4,3,1,""],fit:[4,3,1,""]},"gssl.classifiers.classifier":{GSSLClassifier:[4,1,1,""]},"gssl.classifiers.classifier.GSSLClassifier":{autohooks:[4,3,1,""],fit:[4,3,1,""]},"gssl.classifiers.nn":{NN:[5,0,0,"-"],models:[5,0,0,"-"]},"gssl.classifiers.nn.NN":{Accumulator:[5,1,1,""],NNClassifier:[5,1,1,""],convert_sparse_matrix_to_sparse_tensor:[5,4,1,""],cos_decay:[5,4,1,""],debug:[5,4,1,""],ent:[5,4,1,""],gather:[5,4,1,""],get_S:[5,4,1,""],get_S_fromtensor:[5,4,1,""],kl_divergence:[5,4,1,""],repeat:[5,4,1,""],row_normalize:[5,4,1,""],xent:[5,4,1,""]},"gssl.classifiers.nn.NN.Accumulator":{__init__:[5,3,1,""]},"gssl.classifiers.nn.NN.NNClassifier":{ALPHA:[5,2,1,""],LAMBDA:[5,2,1,""],RECALC_W:[5,2,1,""],SIGMA:[5,2,1,""],USE_UNLABELED:[5,2,1,""],__init__:[5,3,1,""],build_graph:[5,3,1,""],eval_get_data:[5,2,1,""],evaluate_simfunc:[5,3,1,""],fit:[5,3,1,""],labeled_gen:[5,3,1,""],pred_gen:[5,3,1,""],random_gen:[5,3,1,""],unlabeled_gen:[5,3,1,""],unlabeled_pairs_gen:[5,3,1,""]},"gssl.classifiers.nn.models":{conv_large:[5,4,1,""],conv_small:[5,4,1,""],linear:[5,4,1,""],simple:[5,4,1,""]},"gssl.filters":{LDST:[6,0,0,"-"],LGC_LVO:[6,0,0,"-"],MRremoval:[6,0,0,"-"],filter:[6,0,0,"-"],filter_utils:[6,0,0,"-"],ldstRemoval:[6,0,0,"-"]},"gssl.filters.LDST":{LDST:[6,1,1,""]},"gssl.filters.LDST.LDST":{LDST:[6,3,1,""],__init__:[6,3,1,""],fit:[6,3,1,""]},"gssl.filters.LGC_LVO":{LGC_LVO_Filter:[6,1,1,""]},"gssl.filters.LGC_LVO.LGC_LVO_Filter":{LGCLVO:[6,3,1,""],__init__:[6,3,1,""],fit:[6,3,1,""]},"gssl.filters.MRremoval":{MRRemover:[6,1,1,""]},"gssl.filters.MRremoval.MRRemover":{__init__:[6,3,1,""],fit:[6,3,1,""]},"gssl.filters.filter":{GSSLFilter:[6,1,1,""]},"gssl.filters.filter.GSSLFilter":{autohooks:[6,3,1,""],fit:[6,3,1,""]},"gssl.filters.filter_utils":{get_confmat_FN:[6,4,1,""],get_confmat_FP:[6,4,1,""],get_confmat_TN:[6,4,1,""],get_confmat_TP:[6,4,1,""],get_confmat_acc:[6,4,1,""],get_confmat_dict:[6,4,1,""],get_confmat_f1_score:[6,4,1,""],get_confmat_npv:[6,4,1,""],get_confmat_precision:[6,4,1,""],get_confmat_recall:[6,4,1,""],get_confmat_specificity:[6,4,1,""],get_unlabeling_confmat:[6,4,1,""]},"gssl.filters.ldstRemoval":{LDSTRemover:[6,1,1,""]},"gssl.filters.ldstRemoval.LDSTRemover":{LDST:[6,3,1,""],__init__:[6,3,1,""],fit:[6,3,1,""]},"gssl.graph":{gssl_affmat:[7,0,0,"-"],gssl_utils:[7,0,0,"-"]},"gssl.graph.gssl_affmat":{AffMatGenerator:[7,1,1,""],LNP:[7,4,1,""],NLNP:[7,4,1,""],epsilonMask:[7,4,1,""],knnMask:[7,4,1,""],sort_coo:[7,4,1,""]},"gssl.graph.gssl_affmat.AffMatGenerator":{W_from_K:[7,3,1,""],__init__:[7,3,1,""],generateAffMat:[7,3,1,""],get_or_calc_Mask:[7,3,1,""],handle_adaptive_sigma:[7,3,1,""]},"gssl.graph.gssl_utils":{accuracy:[7,4,1,""],accuracy_unlabeled:[7,4,1,""],calc_Z:[7,4,1,""],class_mass_normalization:[7,4,1,""],deg_matrix:[7,4,1,""],extract_lap_eigvec:[7,4,1,""],get_Isomap:[7,4,1,""],get_PCA:[7,4,1,""],get_Standardized:[7,4,1,""],get_pred:[7,4,1,""],init_matrix:[7,4,1,""],init_matrix_argmax:[7,4,1,""],labels_indicator:[7,4,1,""],lap_matrix:[7,4,1,""],scipy_to_np:[7,4,1,""],split_indices:[7,4,1,""]},"input.dataset":{cifar10:[9,0,0,"-"],mnist:[9,0,0,"-"],toy_ds:[9,0,0,"-"]},"input.dataset.cifar10":{ZCA:[9,4,1,""],get_cifar10:[9,4,1,""],load_cifar10_batch:[9,4,1,""],load_cifar10_labelnames:[9,4,1,""],load_cifar10_test:[9,4,1,""]},"input.dataset.mnist":{get_mnist:[9,4,1,""]},"input.dataset.toy_ds":{getDataframe:[9,4,1,""],getTFDataset:[9,4,1,""]},"input.noise":{noise_process:[10,0,0,"-"],noise_utils:[10,0,0,"-"]},"input.noise.noise_process":{LabelNoiseProcess:[10,1,1,""]},"input.noise.noise_process.LabelNoiseProcess":{__init__:[10,3,1,""],corrupt:[10,3,1,""]},"input.noise.noise_utils":{apply_noise:[10,4,1,""],transition_count_mat:[10,4,1,""],uniform_noise_transition_prob_mat:[10,4,1,""]},"log.logger":{LogLocation:[11,1,1,""],debug:[11,4,1,""],error:[11,4,1,""],info:[11,4,1,""],ll:[11,2,1,""],log:[11,4,1,""],set_allowed_debug_locations:[11,4,1,""],warn:[11,4,1,""]},"log.logger.LogLocation":{CLASSIFIER:[11,2,1,""],EXPERIMENT:[11,2,1,""],FILTER:[11,2,1,""],HOOK:[11,2,1,""],MATRIX:[11,2,1,""],NOISE:[11,2,1,""],OTHER:[11,2,1,""],OUTPUT:[11,2,1,""],SPECIFICATION:[11,2,1,""],UTILS:[11,2,1,""]},"output.aggregate_csv":{aggregate_csv:[13,4,1,""],calculate_statistics:[13,4,1,""],debug:[13,4,1,""],info:[13,4,1,""]},"output.folders":{CSV_FOLDER:[13,5,1,""],PLOT_FOLDER:[13,5,1,""],RESULTS_FOLDER:[13,5,1,""],get_top_dir:[13,4,1,""]},"output.plot_core":{authenticate_plotly:[13,4,1,""],color_scale_continuous:[13,4,1,""],color_scale_discrete:[13,4,1,""],plotGraph:[13,4,1,""],vertexplotOpt:[13,1,1,""]},"output.plot_core.vertexplotOpt":{DEFAULT_CONSTANT_COLOR:[13,2,1,""],DEFAULT_UNLABELED_COLOR:[13,2,1,""],__init__:[13,3,1,""]},"output.plots":{plot_all_indexes:[13,4,1,""],plot_labeled_indexes:[13,4,1,""]},experiment:{experiments:[0,0,0,"-"],hooks:[1,0,0,"-"],prefixes:[0,0,0,"-"],selector:[0,0,0,"-"],specification:[2,0,0,"-"]},gssl:{classifiers:[4,0,0,"-"],filters:[6,0,0,"-"],graph:[7,0,0,"-"]},input:{dataset:[9,0,0,"-"],noise:[10,0,0,"-"]},log:{logger:[11,0,0,"-"]},output:{aggregate_csv:[13,0,0,"-"],folders:[13,0,0,"-"],plot_core:[13,0,0,"-"],plots:[13,0,0,"-"]},settings:{ROOT_FOLDER:[14,5,1,""],load_sparse_csr:[14,4,1,""],p_bar:[14,4,1,""],save_sparse_csr:[14,4,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"],"5":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function","5":"py:data"},terms:{"1st":[2,13],"25th":16,"2nd":[2,13],"\u01f9one":4,"boolean":[0,10],"class":[0,1,2,4,5,6,7,10,11,13],"default":[1,4,6,7,13],"disserta\u00e7\u00e3o":16,"enum":[0,11],"float":[0,4,5,6,7,10,13],"function":[0,1,2,4,7,13,16],"import":[4,6],"int":[0,1,4,6,7,10,13],"new":6,"return":[0,2,4,5,6,7,10,13],"sch\u00f6lkopf":16,"static":4,"true":[0,1,2,4,5,6,7,10,13],"try":4,"while":10,For:[2,13],NOT:[6,11],The:[0,1,2,4,5,6,7,10,13,14],There:1,Used:[7,10],Uses:1,Using:16,__init__:[0,1,2,4,5,6,7,10,13],_begin:1,_end:1,aaai:16,about:13,abov:13,abr:[1,2,6],absent:7,absolut:4,accord:[0,13],accumul:5,accur:[0,2,5],accuraci:7,accuracy_unlabel:7,acm:16,actual:10,add:[0,2],add_key_prefix:2,added:[1,2,6],advanc:16,affin:[0,4,5,6,7,13],affmat_cifar10_load:2,affmatconfig:2,affmatgener:[0,7],afo20:[6,16],afonso:16,after:[1,7,10],after_affmat:0,after_classifi:0,after_filt:0,after_nois:0,aggreg:13,aggregate_csv:[2,12,15],alg_it:0,alg_result:0,algconfig:2,algorithm:[0,1,2,4,5,6,7],algorithm_non:2,alia:11,all:[0,2,4,7,13],allpermut:2,alpha:[4,5],alreadi:4,also:[2,6],altern:[0,4,16],analysi:16,andr:16,ani:[0,2],anoth:6,api_kei:13,appear:7,appli:0,apply_nois:10,approach:5,appropri:0,arg:[0,2,5,6,7],argmax:7,argmin:1,argument:[2,4,7],arrai:[0,4,6,7,10,13,14],associ:0,assum:[4,6,7,13],attribut:[1,2,13],augment:5,auth:13,authent:13,authenticate_plotli:13,author:[0,1,2,4,5,6,9,10,11,13,14],autohook:[4,6],automat:[4,6],auxiliari:14,avail:[0,1],balanc:[4,6],base:[0,1,2,4,5,6,7,10,11,13,16],baselin:6,batch_id:9,batch_siz:5,befor:10,before_affmat:0,before_classifi:0,before_filt:0,before_nois:0,begin:[1,4,6],behaviour:4,being:[5,10],belief:[0,1,4,5,6,7,10],belkin2003:[],belkin:16,bernhard:16,best_f1:1,beta:4,better:6,between:[1,4,7,10],bibliographi:15,bn03:[4,16],bool:[0,1,4,6,7,10,13],bousquet:16,bright:13,browser:13,bruno:16,build_graph:5,calc_z:7,calcul:[0,6,7,13],calculate_statist:13,call:[0,1,4,6,10],callback:[0,1],can:0,cannot:6,canva:13,cdist:7,celso:16,certain:0,cfg:2,cfm:16,chanc:10,chang:[4,6,16],change_unlabeled_color:13,chosen:10,cifar10:[8,12],cifar10_matric:2,citat:16,class_mass_norm:7,classdoc:[2,6],classif:[0,1,6,16],classifi:[0,3,11,12],classmethod:[4,6],clean:[0,6,10],clgc:[3,12],clgcclassifi:4,code:2,color:[1,13],color_scale_continu:13,color_scale_discret:13,column:13,com:2,comb:2,combin:[2,4],come:[2,11],comma:13,command:0,complet:10,compos:0,compositehook:1,compris:2,comput:7,confer:16,confid:[6,7],config:[0,2],configur:[0,2,10,13],confus:6,connect:13,consid:[6,7,10],consist:[0,4,6,16],constant:[4,7,13],constantprop:[4,6],constrain:[4,16],construct:7,constructor:[1,4,5,6,10],contain:[0,2,6,7,10,13],content:[12,15],continu:[1,13],control:4,conv_larg:5,conv_smal:5,converg:4,convert_sparse_matrix_to_sparse_tensor:[4,5],coolwarm:13,correct:[0,6,7,10],correspond:[0,1,2,7,13],corrupt:[0,10],corruption_level:2,cos_decai:5,cost:1,could:0,count:10,creat:[0,1,2,4,5,6,7,9,10,11,13],create_video:1,createvideo:1,criteria:6,criterion:4,csr:7,csv:13,csv_folder:13,current:[0,10],data:[1,4,5,6,9,11,13],datafram:13,dataset:[0,2,8,12],dataset_sd:0,debug:[5,11,13],debug_mod:2,decor:2,decreas:[1,7],default_constant_color:13,default_unlabeled_color:13,defin:[2,13],deg_matrix:7,degre:[6,7],deleg:0,dengyong:16,dens:7,depend:7,describ:0,destin:13,detail:7,detect:6,determin:[0,1,4,6,7,10,13],determinist:[2,10],deviat:13,diagnosi:0,diagon:7,dict:[0,2],dict_a:2,dict_b:2,dictionari:[0,2],differ:[1,10,13],digit1:[],dimens:[0,4,5,6,7],directli:6,directori:13,disabl:11,discourag:6,discret:[1,13],dispers:0,displai:1,dist:7,dist_func:[2,7],distanc:7,distr_1:5,distr_2:5,distribut:10,document:[0,7],doe:2,doi:16,doid:16,down:0,ds_name:9,dtype:5,dure:[4,5,6,10],during_affmat:0,during_classifi:0,during_filt:0,during_nois:0,each:[0,1,2,6,7,10,13],early_stop:6,eclips:[2,13,14],edg:[4,5,6,7,13],edge_mat:4,edge_width:13,eigenbasi:4,eigenfunct:4,eigenmap:4,eigenvalu:7,eigenvector:[4,7],either:[1,4,13],empti:2,emptyspecif:2,encapsul:[0,13],encod:[0,4,5,6,10],end:[1,4,6],ent:5,entri:7,enumer:0,epoch_var:5,eps:7,epsilon:7,epsilonmask:7,equal:[6,7],equiprob:[4,6],error:[10,11],essenti:10,estim:[4,6,7],estimatedfreq:7,euclidean:7,eval_get_data:5,evaluate_simfunc:5,everi:[0,2,7],exactli:2,exampl:[1,2],execut:[0,1,4,5,6,13],exp:7,exp_chapel:[0,12],exp_cifar:[0,12],exp_debug:[0,12],exp_filter_ldst:[0,12],expchapel:2,expchapelle_2:2,expchapelle_3:2,expchapelle_4:2,expchapelle_5:2,expcifar:2,expdebug:2,expect:[2,10,13],experi:[5,11,12,13,15],extend:2,extens:[0,1,13],extra:[4,5,6,7],extract:[7,13],extract_lap_eigvec:7,f_0:4,factor:4,fals:[1,2,4,5,6,7,9,13],featur:0,few:13,field:[0,4,16],file:[1,13],filenam:[0,1,13,14],filename_path:1,files_to_join:13,filter:[0,1,2,3,11,12],filter_aft:0,filter_it:0,filter_mr:2,filter_nofilt:2,filter_util:[3,12],filterconfig:2,filterldst:2,first:[2,4],fit:[4,5,6],fix:10,flag:11,flat:7,flatten:9,flip:10,float32:5,folder:[12,14,15],follow:[0,10],forc:11,force_gtam_ldst_same_mu:2,force_lb_callback:1,force_y_callback:1,forest:4,form:1,forward:0,found:[0,10],frac:[4,10],freq:[4,6],frequenc:[4,6,7],from:[0,1,2,6,7,10,11,13],fun:[4,6],func:2,futur:1,g241c:2,gather:[4,5],gaussian:[0,2,4,7,16],gener:[0,2,7,13],generalconfig:2,generateaffmat:7,get:[0,1,2,6,7,13],get_:5,get_all_config:2,get_cifar10:9,get_confmat_acc:6,get_confmat_dict:6,get_confmat_f1_scor:6,get_confmat_fn:6,get_confmat_fp:6,get_confmat_npv:6,get_confmat_precis:6,get_confmat_recal:6,get_confmat_specif:6,get_confmat_tn:6,get_confmat_tp:6,get_isomap:7,get_mnist:9,get_or_calc_mask:7,get_p:4,get_pca:7,get_pr:7,get_s_fromtensor:[4,5],get_spec_nam:2,get_standard:7,get_step_r:1,get_top_dir:13,get_unlabeling_confmat:6,getdatafram:9,gettfdataset:9,gfhf:[0,3,12],gfhflassifi:0,ghahramani:16,gist:2,github:2,given:[0,2,4,6,7,10,13],global:[0,4,6,16],good:4,gradient:1,gradient_fix:6,graph:[0,3,4,12,13,16],group:13,gssl:[0,1,12,13,15],gssl_affmat:[0,3,12],gssl_util:[3,12],gsslclassifi:[0,4,5],gsslfilter:[0,6],gsslhook:[0,1,4,5,6,10],gtam:[0,1,3,7,12],gtam_f:0,gtam_i:0,gtam_q:0,gtamclassifi:[0,4],handl:7,handle_adaptive_sigma:7,harmon:[0,4,16],has:[6,7],have:[1,2,6,10,11,13],help:[0,2,5],higher:[6,16],highli:6,hist:[],home:[2,13,14],hook:[0,4,5,6,7,10,11,12],hook_list:[0,1],hook_mod:0,hook_skeleton:[0,12],hooktim:0,hot:[0,10],how:[1,10],http:[2,16],icml:16,identifi:[0,2,11],iff:7,ignor:5,imag:[1,13],image_shap:5,inadequ:4,includ:[1,10,13],inclus:4,incorrect:4,independ:10,index:[6,7,13,15],indic:[0,6,7,10,13],individu:13,infer:0,influenc:6,info:[11,13],inform:[0,13,16],init:7,init_al:0,init_label:0,init_matrix:7,init_matrix_argmax:7,init_v:5,initi:[0,1,2,4,5,10,13],input:[0,2,4,5,6,7,12,15],input_shap:5,inputconfig:2,instanc:[0,1,4,5,6,7,10],instead:[6,7],intend:5,interact:13,intern:16,interpret:[0,6,10],inv_norm:7,invalid:[7,10],is_norm:7,isolet:2,iter:[1,4,6],itself:2,jason:16,jebara:16,john:16,join:[2,13],jun:[4,16],keep:[1,6],keep_imag:1,kei:[0,1,2,10],kept:6,keyerror:[0,10],keys_multiplex:0,kind:13,kl_diverg:5,klau:[0,1,2,4,5,6,9,10,11,13,14,16],knn:7,knnmask:7,know_true_freq:[4,6],kwarg:[0,1,2,6,7,10],label:[0,1,4,5,6,7,10,13,16],label_se:9,labeled_gen:5,labeled_index:7,labeled_onli:13,labeled_perc:0,labeledindex:[1,4,5,6,7,10,13],labelnoiseprocess:[0,10],labels_ind:7,lafferti:16,lal:16,lambda:5,lap_matrix:7,lapeigl:[3,12],laplacian:[4,7],lb_f:6,lb_n:6,ldst:[0,3,7,12],ldst_filterstats_hook:[0,12],ldst_stats_hook:0,ldstfilterstatshook:1,ldstremov:[3,12],learn:[4,16],learning_r:5,least:4,leav:6,lgc:[0,3,12],lgc_iter_tf:4,lgc_lvo:[3,12],lgc_lvo_filt:6,lgc_tf:[3,12],lgcclassifi:[0,4],lgclvo:6,lgclvo_f:6,line:13,linear:[4,5,7],link:2,list:[0,1,2,11,13],lnp:7,load:[2,7,13],load_cifar10_batch:9,load_cifar10_labelnam:9,load_cifar10_test:9,load_path:[2,7],load_sparse_csr:14,loc:11,local:[0,4,6,16],locat:[11,13],log:[12,15],log_loc:11,log_typ:11,logger:[12,15],logloc:11,lower:1,machin:16,mai:[0,10,11],main:0,maintain:4,make:4,make_blob:0,make_moon:0,mani:[1,7,10],manifold:[4,6,16],manifoldreg:[],map:2,mar:[0,2,4,10,13],mark:[0,1,6,13],mask:7,mask_func:[2,7],master:16,math:10,matric:7,matrix:[0,1,4,5,6,7,10,11,13],max:7,max_it:4,max_valu:14,maximum:13,mean:[2,10,13],measur:0,median:13,merg:13,messag:11,mestrado:16,method:[1,2,4,6],metric:7,might:[2,10],min:7,minim:[0,4,7,16],minimum:13,mit:16,mnist:[2,8,12],mode:[1,7,13],model:[3,4],model_choic:5,modifi:[1,6],modul:12,more:[0,1,6,7],mp4:1,mplex:0,mrf:2,mrremov:[3,12],msg:[5,11,13],multipl:13,multiplex:0,multipli:4,must:[1,13],mut:7,n_estim:4,n_neighbor:7,na_replace_v:7,name:[0,1,2,5],nan:1,navin:16,ncar:[2,10],ndarrai:[0,4,5,6,7,10,13],nearest:7,neighbor:7,neighborhood:7,network:5,neural:[5,16],niyogi:16,nlnp:7,nnclassifi:5,nois:[0,2,8,11,12,16],noise_aft:0,noise_corrupt:10,noise_process:[0,8,12],noise_uniform_det_moder:2,noise_util:[8,12],noiseconfig:2,noisi:[0,6,10],noisygssl:[2,13,14],non:7,none:[0,1,2,4,5,6,7,10,13],nonzero:7,norm:4,normal:[6,7],normalize_row:[6,7],nov:[4,6,9],now:5,npz:2,num_anchor:7,num_epoch:5,num_it:4,num_label:9,num_palett:13,number:[4,6,7,10,13],numpi:5,object:[0,1,2,4,5,6,7,10,13],obtain:[0,10,13],occur:[1,7],occurr:10,often:7,olivi:16,onc:6,one:[0,2,6,7,10,13],ones:7,onli:[7,13],onlin:13,only_label:1,open:13,oper:[1,4,5,6,13],optim:5,option:[0,1,4,5,6,7,10,13],order:[7,16],org:16,origin:[2,10,13],other:[1,6,10,11],otherwis:[1,6,7],out:[2,5,6],out_siz:5,output:[0,1,11,12,15],output_dict:1,output_path:13,output_prefix:13,output_shap:5,overrid:1,overwrit:2,p_bar:14,packag:[12,15],page:[15,16],pair:[0,7,10],palett:[1,13],pallet:13,parallel:2,param:7,paramet:[0,1,2,4,5,6,7,10,11,13],part:1,partial:[0,16],pass:[4,6],path:[1,7,13,14],paulo:16,pct_leq:[],percentag:[0,4,6,7,10],perform:[0,4,6],permut:2,phd:16,pick:[2,6],pickabl:2,place:0,pleas:5,plot:[0,1,4,5,6,12,15],plot_all_index:13,plot_cor:[12,15],plot_filepath:13,plot_fold:13,plot_hook:[0,12],plot_id:1,plot_labeled_index:13,plot_mod:1,plot_siz:13,plotgraph:13,plothook:1,plotitergtamhook:1,plotiterhook:1,plotli:13,point:[0,13],portal:16,posit:[7,13],possibl:[2,13],postprocess:0,precalcul:7,precis:6,pred_gen:5,predict:7,pref:2,prefix:[2,12,13,15],press:16,previous:[5,7],print:13,probabl:10,problem:7,procedur:[1,13],proceed:16,process:[0,2,4,10,16],processifi:2,produc:[2,13],project:14,propag:[4,6,7],propagaton:7,properti:[4,13],proport:[4,6],purpos:[0,10],pwr:7,python:14,python_plotli:13,quantiti:10,rais:[0,7,10,13],rampdown_length:5,random:[4,10],random_gen:5,raw_ann_k:2,reach:6,read:0,reason:4,recalc_w:5,reciproc:7,recommend:6,reconstruct:7,reflect:10,reg:9,regular:[4,6,16],relabel:6,relat:[6,10,13],relev:[0,6,14],remain:7,remov:6,remove_first_eig:7,repeat:[4,5],repres:2,reproduc:[0,7,10],requir:[0,4,7,10],respect:13,restrict:4,result:[4,7,10,13],results_fold:13,rfclassifi:4,rho:4,rmfolder:1,rodrigu:16,root:14,root_fold:14,row:[0,1,6,7,10],row_norm:[4,5],run:[0,2,4,10,13],run_al:2,run_debug_example_al:0,run_debug_example_on:0,runprocessify_func:2,runtim:0,sacrif:6,same:[5,6,13],sampl:7,save:1,save_sparse_csr:14,scale:10,schlamar:2,scipi:7,scipy_to_np:7,search:15,second:2,see:[0,2,4,5,6,7],seed:[0,7,10],select:[6,10],select_affmat:0,select_and_add_hook:0,select_classifi:0,select_filt:0,select_input:0,select_nois:0,selector:[12,15],self:[0,2,5],semi:[4,16],separ:13,set:[1,2,7,9,10,11,12,13,15],set_allowed_debug_loc:11,shape:[0,4,5,6,7,10,13],share:13,shih:16,should:[1,4,6,10,11],show:13,showcas:1,sigma:[2,5,7],signatur:[0,2,5],sii:[3,12],siisclassifi:4,simpl:5,simpli:0,size:[2,13],sk_gaussian:0,sk_spiral:0,skeleton:[1,4,6],sklearn:0,slideshow:1,slowdown_factor:1,smaller:13,smooth:4,some:[0,1,5,7],sort_coo:7,sou17:16,sousa:16,sousaconstrain:4,spatial:7,specif:[0,1,6,7,10,11,12],specifi:[0,1,2,7,10,13],specification_bit:[0,12],specification_skeleton:[0,12],split:7,split_indic:7,split_p:7,squar:4,src:[2,14,15],stage:2,standard:13,stat:1,statist:13,step:[0,1,6],step_siz:1,store:[1,14],str:[1,7,13],stratif:7,stratifi:7,string:13,structur:[13,16],subfold:1,submodul:[3,8,12,15],subpackag:[12,15],sum:[6,7],summari:13,supervis:[4,16],support:7,suppress:11,sure:2,swap:10,sym:7,symm:7,symmetr:7,system:16,t_affmat:0,t_alg:0,t_filter:0,t_nois:0,take:[0,1,2],taken:[0,1],target:10,temp_subfolder_nam:1,temporari:1,tensor:5,term:[4,6],thesi:16,thi:[0,1,2,5,10,11,13,14],thick:13,thoma:16,thresh_otsu:[],through:[0,4,7],time:[0,1,5],time_hook:[0,12],timehook:1,timer:1,timer_nam:1,titl:[1,13],title_text:14,toi:0,toni:16,total:6,total_it:4,toy_d:[8,12],transduct:[0,4,16],transit:10,transition_count_mat:10,tri:10,tune:[0,6],tuning_it:[2,6],tuning_iter_as_noise_pct:2,tuning_iter_as_pct:[2,6],tupl:0,twentieth:16,two:2,txt:13,type:[0,1,2,4,5,6,7,10,13],under:4,unifesp:16,uniform:10,uniform_noise_transition_prob_mat:10,union:[4,6,13],universidad:16,unlabel:[1,5,7,13],unlabeled_gen:5,unlabeled_pairs_gen:5,unlabeled_size_multipli:13,unless:4,unnorm:7,updat:[1,4,5,6],update_f:4,url:16,usag:13,use:[0,4,6,7,13],use_baselin:6,use_unlabel:5,useconstantprop:4,used:[0,1,7,10],useestimatedfreq:[4,6],uselgcmat:[],usernam:13,uses:[4,6,13],usez:6,using:[4,5,7,16],util:[6,7,10,11],valu:[0,1,2,4,7,10,13],valueerror:[7,10,13],var_valu:[4,5],variabl:[1,10,13],vector:7,version:[6,10],vertex:[7,13],vertex_opt:13,vertexoptobject:13,vertexplotopt:13,vertic:[6,13],via:[0,16],video:1,video_path:1,w_filter_aft:0,w_from_k:7,w_init_al:0,w_init_label:0,w_noise_aft:0,w_sparse_v:5,wai:[4,6],wang:16,warn:[7,11],weigh_by_degre:[4,6,7],weight:[4,5,6,7],welf_sample_var:[],well:0,weston:16,when:[1,2,6,7,13],whenev:6,where:[0,7,11],whether:[1,7,13],which:[0,1,6,7,10,11,13],whose:[7,10],wise:7,within:4,wjc08:[4,16],workspac:[2,13,14],write_freq:2,x_i:7,x_j:7,xent:5,xiaojin:16,y_f:6,y_n:6,y_pred:7,y_true:[5,6,7],zbl:[4,16],zca:9,zgl03:[4,16],zhou:16,zhu:16,zoubin:16},titles:["experiment package","experiment.hooks package","experiment.specification package","gssl package","gssl.classifiers package","gssl.classifiers.nn package","gssl.filters package","gssl.graph package","input package","input.dataset package","input.noise package","log package","src","output package","settings module","NoisyGSSL Module","Bibliography"],titleterms:{aggregate_csv:13,bibliographi:16,cifar10:9,classifi:[4,5],clgc:4,content:[0,1,2,3,4,5,6,7,8,9,10,11,13],dataset:9,exp_chapel:2,exp_cifar:2,exp_debug:2,exp_filter_ldst:2,experi:[0,1,2],filter:6,filter_util:6,folder:13,gfhf:4,graph:7,gssl:[3,4,5,6,7],gssl_affmat:7,gssl_util:7,gtam:4,hook:1,hook_skeleton:1,indic:15,input:[8,9,10],lapeigl:4,ldst:6,ldst_filterstats_hook:1,ldstremov:6,lgc:4,lgc_lvo:6,lgc_tf:4,log:11,logger:11,manifoldreg:[],mnist:9,model:5,modul:[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15],mrremov:6,nois:10,noise_process:10,noise_util:10,noisygssl:15,output:13,packag:[0,1,2,3,4,5,6,7,8,9,10,11,13],plot:13,plot_cor:13,plot_hook:1,prefix:0,selector:0,set:14,sii:4,specif:2,specification_bit:2,specification_skeleton:2,src:12,submodul:[0,1,2,4,5,6,7,9,10,11,13],subpackag:[0,3,4,8],tabl:15,time_hook:1,toy_d:9}})