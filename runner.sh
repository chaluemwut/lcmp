#!/bin/sh
python cmp.py bagging heart &
python cmp.py bagging letter &
python cmp.py bagging austra &
python cmp.py bagging german &
python cmp.py bagging sat &
python cmp.py bagging segment &
python cmp.py bagging vehicle &
python cmp.py boosted heart &
python cmp.py boosted letter &
python cmp.py boosted austra &
python cmp.py boosted german &
python cmp.py boosted sat &
python cmp.py boosted segment &
python cmp.py boosted vehicle &
python cmp.py randomforest heart &
python cmp.py randomforest letter &
python cmp.py randomforest austra &
python cmp.py randomforest german &
python cmp.py randomforest sat &
python cmp.py randomforest segment &
python cmp.py randomforest vehicle &
python cmp.py knn heart &
python cmp.py knn letter &
python cmp.py knn austra &
python cmp.py knn german &
python cmp.py knn sat &
python cmp.py knn segment &
python cmp.py knn vehicle &
python cmp.py svm heart &
python cmp.py svm letter &
python cmp.py svm austra &
python cmp.py svm german &
python cmp.py svm sat &
python cmp.py svm segment &