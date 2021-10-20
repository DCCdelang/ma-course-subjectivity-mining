import logging
import sys

from tasks import vua_format as vf
from ml_pipeline import utils, cnn, preprocessing, pipeline_with_lexicon
from ml_pipeline import pipelines, representation
from ml_pipeline.cnn import CNN, evaluate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
#handler = logging.FileHandler('experiment.log')
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def run(task_name, data_dir, pipeline_name, print_predictions, GridSearch, ImpFea, conf_matrix):
    logger.info('>> Running {} experiment'.format(task_name))
    tsk = task(task_name)
    logger.info('>> Loading data...')
    tsk.load(data_dir)
    logger.info('>> retrieving train/data instances...')
    train_X, train_y, test_X, test_y = utils.get_instances(tsk, split_train_dev=False)
    test_X_ref = test_X

    if pipeline_name.startswith('cnn'):
        pipe = cnn(pipeline_name)
        train_X, train_y, test_X, test_y = pipe.encode(train_X, train_y, test_X, test_y)
        logger.info('>> testing...')
    else:
        pipe = pipeline(pipeline_name)
  
    logger.info('>> training pipeline ' + pipeline_name)
    pipe.fit(train_X, train_y)
    if pipeline_name == 'naive_bayes_counts_lex':
        logger.info("   -- Found {} tokens in lexicon".format(pipe.tokens_from_lexicon))

    logger.info('>> testing...')

    if GridSearch == True:
        if "svm" in pipeline_name:
            params = pipelines.svm_clf_grid_parameters()
            sys_y = utils.grid_search(pipe, params , train_X, train_y, test_X)
        else:
            print("Gridsearch not available, will perform normal predict.")
            sys_y = pipe.predict(test_X)
    else:
        sys_y = pipe.predict(test_X)

    logger.info('>> evaluation...')
    logger.info(utils.eval(test_y, sys_y))
    if ImpFea == True:
        if "svm" not in pipeline_name:
            if "cnn" not in pipeline_name:
                if "lex" not in pipeline_name:
                    utils.important_features_per_class(pipe.named_steps.frm, pipe.named_steps.clf)
        if "svm" in pipeline_name:
            utils.important_features_per_class_SVM(pipe.named_steps.frm.fit(train_X), pipe.named_steps.clf)
        


    if print_predictions:
        logger.info('>> predictions')
        logger.info(utils.print_all_predictions(test_X_ref, test_y, sys_y, logger))
    if conf_matrix:
        logger.info(">> plotting confusion matrix")
        logger.info(utils.print_error_analysis(pipe,test_X_ref,test_y,sys_y,logger))


def task(name):
    if name == 'vua_format':
        return vf.VuaFormat()
    else:
        raise ValueError("task name is unknown. You can add a custom task in 'tasks'")


def cnn(name):
    if name == 'cnn_raw':
        return CNN()
    elif name == 'cnn_prep':
        return CNN(preprocessing.std_prep())
    else:
        raise ValueError("pipeline name is unknown.")


def pipeline(name):
    if name == 'naive_bayes_counts':
        return pipelines.naive_bayes_counts()
    elif name == 'naive_bayes_tfidf':
        return pipelines.naive_bayes_tfidf()
    elif name == 'naive_bayes_counts_lex':
        return pipeline_with_lexicon.naive_bayes_counts_lex()
    elif name == 'svm_libsvc_counts':
        return pipelines.svm_libsvc_counts()
    elif name == 'svm_libsvc_tfidf':
        return pipelines.svm_libsvc_tfidf()
    elif name == 'svm_libsvc_embed':
        return pipelines.svm_libsvc_embed()
    elif name == 'svm_sigmoid_embed':
        return pipelines.svm_sigmoid_embed()
    elif name == 'svm_libsvc_tfidf_char':
        return pipelines.svm_libsvc_tfidf_char()
    elif name == 'naive_bayes_counts_lex':
        return pipelines.naive_bayes_counts_lex()
    elif name == 'naive_bayes_counts_bigram':
        return pipelines.naive_bayes_counts_bigram()
    elif name == 'svm_libsvc_tfidf_lem':
        return pipelines.svm_libsvc_tfidf_lem()
    else:
        raise ValueError("pipeline name is unknown. You can add a custom pipeline in 'pipelines'")

