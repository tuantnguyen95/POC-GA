import traceback
import argparse
import joblib
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import Perceptron
from data import DataGenerator
from image_feature_generator import ImageGenerator
from sk_utils import str_to_bool


def main(args):
  visual_gen = None
  if args.use_visual_sim:
    visual_gen = ImageGenerator(args.visual_model_dir, size=160)
  train_gen = DataGenerator(
    args.train_csv, visual_gen=visual_gen,
    raw_path=args.raw_path,
    use_visual_sim=args.use_visual_sim,
    use_xpath_sim=args.use_xpath_sim,
    use_ocr_sim=args.use_ocr_sim,
    use_classname_sim=args.use_classname_sim,
    use_id=args.use_id, use_text=args.use_text,
    use_recur_text=args.use_recur_text)

  test_gen = DataGenerator(
    args.test_csv, visual_gen=visual_gen,
    raw_path=args.raw_path,
    use_visual_sim=args.use_visual_sim,
    use_xpath_sim=args.use_xpath_sim,
    use_ocr_sim=args.use_ocr_sim,
    use_classname_sim=args.use_classname_sim,
    use_id=args.use_id, use_text=args.use_text,
    use_recur_text=args.use_recur_text)

  clfs = {
    'svm': svm.SVC(gamma='scale', kernel='rbf'),
    'svm_lin': svm.SVC(kernel="linear", C=0.025),
    'svm_gama_c': svm.SVC(gamma=2, C=1),
    'naive-bayes' : GaussianNB(),
    'BernoulliNB': BernoulliNB(),
    'MultinomialNB': MultinomialNB(),
    'RandomForest': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    'AdaBoost': AdaBoostClassifier(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    'perceptron': Perceptron(random_state=0),
    'DecisionTreeClassifier': tree.DecisionTreeClassifier(),
    'MLPClassifier' : MLPClassifier(solver='lbfgs', alpha=1e-5,
                                    hidden_layer_sizes=(5, 2), random_state=1)}

  for name, clf in clfs.items():
    try:
      print('=================START %s ===============\n\n\n\n' %(name,))
      clf.fit(train_gen.data, train_gen.target)
      filename = name+'.joblib.pkl'
      _ = joblib.dump(clf, filename, compress=9)
      clf = joblib.load(filename)
      y_score = clf.predict(test_gen.data)
      average_precision = average_precision_score(test_gen.target, y_score)
      print(classification_report(test_gen.target, y_score, labels=[1, -1]))
      print('Average precision-recall score: {0:0.2f}'.format(average_precision))
      if name == 'perceptron':
        print('coefficients:', clf.coef_)
      print('=================END===============\n\n\n\n')
    except Exception as e:
      traceback.print_exc()
      print(e)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('train_csv', help='path to the train csv file', type=str)
  parser.add_argument('test_csv', help='path to the test csv file', type=str)
  parser.add_argument('raw_path',
                      help='path to dir containing elements and screenshots', type=str)
  parser.add_argument('visual_model_dir',
                      help='path to our facenet pretrain model', type=str)
  parser.add_argument('--use_visual_sim', default=True, type=str_to_bool,
                      help='specify whether visual features are used')
  parser.add_argument('--use_xpath_sim', default=True, type=str_to_bool,
                      help='specify whether xpath features are used')
  parser.add_argument('--use_ocr_sim', default=True, type=str_to_bool,
                      help='specify whether ocr features are used')
  parser.add_argument('--use_classname_sim', default=True, type=str_to_bool,
                      help='specify whether classname features are used')
  parser.add_argument('--use_id', default=True, type=str_to_bool,
                      help='specify whether resource-id or accesibility-id are used as features')
  parser.add_argument('--use_text', default=True, type=str_to_bool,
                      help='specify whether the text/label attributes are used as features')
  parser.add_argument('--use_recur_text', default=True, type=str_to_bool,
                      help='specify whether the recursive attributes are used as features')

  args = parser.parse_args()
  print('args: ', args)
  main(args)
