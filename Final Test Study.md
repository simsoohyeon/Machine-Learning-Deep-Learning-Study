## 6. 결정 트리 

### 6.2 예측 (1)
```
- 노드의 속성
1. sample 속성: 얼마나 많은 훈련 샘플이 적용되었는지 계산
2. value 속성: 노드에서 각 클래스에 얼마나 많은 훈련 샘플이 있는지 계산
3. gini 속성: 지니 불순도 계산
```
![image](https://github.com/simsoohyeon/Machine-Learning-Deep-Learning-Study/assets/127268889/dd3cbbec-cf17-4fdd-a400-55995a1ab89e)
```
지니 불순도란, 결정 트리에서 사용되는 노드의 불순도를 측정하는 지표
불순도는 노드가 얼마나 혼합되어있는지를 나타내며, 지니 불순도가 낮을수록 해당 노드는 더 '순수한' 것이다
순수한 노드는 대부분 하나의 클래스에 속한 샘플들로 이루어져 있다
```
#### 모델 해석: 화이트 박스와 블랙 박스
```
- 화이트 박스 모델: 결정 트리와 같이 직관적이고 결정 방식을 이해하기 쉬운 모델
- 블랙 박스 모델
: 랜덤 포레스트나 신경망 등의 알고리즘 모델, 성능이 뛰어나고 예측 만드는 연산 과정 쉽게 확인 가능
but 왜 그런 예측을 만드는지는 설명하기 어려움
- 결정 트리는 필요하다면 수동으로 직접 따라 해볼 수 있는 간단하고 명확한 분류 방법 사용
  - 해석 가능한 ML(interpretable ML) 분야는 사람이 이해할 수 있는 방식으로 모델의 결정 설명할 수 있는 ML 시스템 만드는 것을 목표
  - 이는 시스템이 불공정한 결정을 내리지 않도록 하는 많은 영역에서 중요
```
### 6.7 규제 매개변수
```
from sklearn.datasets import make_moons
-> sklearn.datasets 모듈에서 make_moons 함수 불러오기, 이 함수는 두 개의 반달 모양 데이터 포인트 생성하는데 사용

X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)
-> 함수 이용하여 150개의 샘플로 구성된 데이터셋 생성
noise=0.2 는 약간의 잡음을 추가하며, 난수 생성기의 시드를 설정하여 재현성을 보장함
X_moons는 특성으로 구성된 배열이고, y_moons는 타겟 레이블로 구성된 배열

tree_clf1 = DecisionTreeClassifier(random_state=42)
-> 첫 번째 결정트리 분류기 생성, 결과의 재현성 보장

tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
-> 두 번째 결정 트리 분류기 생성,min_sample_left=5는 리프 노드에 최소 5개의 샘플이 있어야 한다는 규제 조건 추가

tree_clf1.fit(X_moons, y_moons)
-> 첫 번째 결정 트리 분류기를 X_moons와 y_moons 데이터에 맞춰 학습

tree_clf2.fit(X_moons, y_moons)
-> 두 번째 결정 트리 분류기를 X_moons와 y_moons 데이터에 맞춰 학습
```
![image](https://github.com/simsoohyeon/Machine-Learning-Deep-Learning-Study/assets/127268889/300b4c5c-b988-48aa-9a92-11a3544a51f7)
```
그림은 두 개의 결정 트리 모델의 결정 경계를 시각화한 것
1. 왼쪽 그림(규제 없음)
: 첫 번째 결정트리 모델의 경계로 규제를 사용하지 않아 데이터에 매우 잘 맞추려 하기 때문에
결정 경계가 매우 복잡하고 세부적인 패턴을 많이 따름, 데이터의 잡음과도 잘 맞추려 하기 때문에
과적합이 될 가능성이 높음 -> 새로운 데이터에 대해 일반화 성능 떨어짐
2. 오른쪽 그림(규제 있음)
: 두 번째 결정트리 모델의 경계로, 리프 노드에 최소 5개의 샘플이 있어야 하는 규제 사용
결정 경계가 더 단순하고 부드럽게 나타남, 이는 모델이 과적합을 피하고 더 잘 일반화할 수 있도록 도움
노드에 최소 샘플 수를 설정함으로써, 모델은 덜 복잡해지고 더 안정적인 예측 가능해져 일반화 성능 향상

결론: 결정 트리 모델에 규제를 추가함으로써 과적합을 줄이고 모델의 일반화 성능을 개선할 수 있음을 확인
```

```
다른 랜덤 시드로 생성한 테스트 세트에서 두 결정 트리를 평가
X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2, random_state=43
-> make_moons 함수 사용하여 테스트용 데이터셋을 생성, 이 데이터셋은 1000개의 샘플로 구성
약간의 잡음 추가, 난수 생성기 통해 결과의 재현성 보장, 특성과 타겟 레이블로 구성된 배열

tree_clf1.score(X_moons_test, y_moons_test)
-> 첫 번째 결정 트리 모델의 테스트 데이터셋에 대한 정확도를 평가함
score 메소드는 주어진 특성과 타겟 데이터에 대해 모델의 정확도 반환함

tree_clf2.score(X_moons_test, y_moons_test)
-> 두 번째 결정 트리 모델의 테스트 데이터셋에 대한 정확도 평가함
```

### 6.8 회귀 (1)
```
사이킷런의 DecisionTreeRegressor를 사용해 잡음이 섞인 2차 함수 형태의 데이터셋에서
max_depth=2 설정으로 회귀 트리 만들기

import numpy as np
from sklearn.tree import DecisionTreeRegressor
-> numpy 라이브러리를 np로 임포트 하고, sklearn.tree 모듈에서 클래스 임포트
DecisionTreeRegressor는 결정 트리를 사용한 회귀 모델을 만드는데 사용됨

np.random.seed(42)
-> 난수 생성기로 시드를 42로 설정하여 결과의 재현성 보장함
이렇게 하면 코드가 실행될 때마다 동일한 난수르 생성하게 됨

X_quad = np.random.rand(200, 1) - 0.5  # 랜덤한 하나의 입력 특성
-> 200개의 샘플을 가지고 하나의 특성으로 이루어진 랜덤한 입력 데이터를 생성함
값의 범위는 -0.5부터 0.5 사이임

y_quad = X_quad ** 2 + 0.025 * np.random.randn(200, 1)
-> 출력 값을 생성함, 입력값 제곱에 약간의 잡음을 더한 값임, 잡음의 평균은 0이며 표준편차가 0.025

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
-> 결정트리 회귀 모델 생성, 트리의 최대 깊이를 2로 제한함, 결과의 재현성 보장

tree_reg.fit(X_quad, y_quad)
-> 결정 트리 회귀 모델을 생성한 데이터 X_quad와 y_quad에 맞춰 학습 시킴
```
### 6.8 회귀 (2)
![image](https://github.com/simsoohyeon/Machine-Learning-Deep-Learning-Study/assets/127268889/088f8ae8-bb0d-4563-8e9e-40cb8476694f)
```
- 각 영역의 예측값은 항상 그 영역에 있는 타깃값의 평균값이 됨
- 알고리즘은 예측값과 가능한 한 많은 샘플이 있도록 영역을 분할

- max_depth=2
: 단순한 모델로, 데이터의 전반적인 패턴을 따르지만 복잡한 패턴을 놓칠 수 있음
과적합의 위험은 적지만 과소적합이 될 가능성이 있음
-max_depth=3
: 더 보잡한 모델로, 데이터의 세부적인 변동을 더 잘 캡처함
과적합의 위험은 있지만 적절한 규제를 통해 이를 방지할 수 있음

두 그래프 모두 결정 트리의 깊이에 따라 예측이 어떻게 변화하는지 보여준다
더 깊은 트리는 복잡한 패턴을 학습할 수 있지만 과적합의 위험도 증가시킨다
따라서 결정 트리 모델의 깊이는 데이터의 복잡성과 일반화 성능을 고려하여 적절히 선택해야 함
분류에서와 같이 회귀 작업에서도 결정 트리가 과대적합되기 쉬우므로 규제가 필요
```

### 6.9 축 방향에 대한 민감성 (1)
```
- 결정트리는 계단 모양의 결정 경계를 만듦, 모든 분할은 축에 수직 (데이터의 방향에 민감)
=> 훈련 세트의 회전에 따라 결정트리의 축이 변할 수 있음
```

### 6.9 축 방향에 대한 민감성 (2)
```
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
-> PCA 주성분 분석을 위한 클래스
-> make_pipeline 여러 변환기 transformer와 추정기 estimator를 연결하여 파이프라인 만드는 함수
-> StandardScaler 데이터를 표준화(평균을 0, 분산을 1)로 하는 변환기

pca_pipeline = make_pipeline(StandardScaler(), PCA())
-> StandardScaler와 PCA를 순차적으로 적용하는 파이프라인 생성
이 파이프라인은 먼저 데이터를 표준화한 후 주성분 분석을 수행함

X_iris_rotated = pca_pipeline.fit_transform(X_iris)
-> X_iris 데이터를 파이프라인에 맞춰 변환함, 데이터는 먼저 표준화한 후 주성분 분석을 통해 회전
X_iris_ratated는 변환된 데이터임

tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
-> 최대 깊이가 2로 제한된 결정 트리 분류기를 생성, 결과의 재현성 보장

tree_clf_pca.fit(X_iris_rotated, y_iris)
-> 변환된 데이터와 타겟값 y_iris를 사용해 결정 트리 분류기 학습시킴

<결과 해석>
- 다른 색과 모양의 점들은 각각 다른 붗꽃 품종을 나타낸다
- 결정 트리가 각 클래스를 분류하기 위해 학습한 경계를 시각화한 것이다 깊이가 2로 제한
- 깊이 0에서 처음 분할이 이루어지고, 깊이 1에서 두 번째 분할이 이루어짐
- PCA를 통해 데이터의 차원을 축소하고 회전함으로써 결정트리가 각 클래스를 분류하는데 주요 특성을 쉽게 식벼
- 이 결과는 결정 트리 모델이 PCA로 변환된 데이터에서 잘 작동할 수 있음을 시사
```

### 6.10 결정 트리의 분산 문제
```
결정 트리의 주요 문제는 분산이 상당히 큼
-> 즉, 하이퍼파라미터나 데이터를 조금만 변경해도 매우 다른 모델이 생성될 수 있음
```

## 7. 앙상블 학습과 랜덤 포레스트 
### 7.1 투표 기반 분류기 (1)
```
- 앙상블 학습
: 일련의 예측기(분류나 회귀모델)로부터 예측을 수집하면 가장 좋은 모델 하나보다 더 좋은 예측을 얻을 수 있을 것임
랜덤 포레스트는 결정 트리의 앙상블

- 투표 기반 분류기
: 정확도가 80%이상인 분류기를 여러 개 훈련 시켰다고 가정
로지스틱 회귀 분류기, SVM 분류기, 랜덤 포레스트 분류기, K-최근접 이웃 분류기 등

- 직접 투표 분류기 hard voting
: 강한 학습기, 약한 학습기
```
### 7.1 투표 기반 분류기 (4)
```
from sklearn.datasets import make_moons
-> 모듈에서 make_moons 함수 불러움, 이 함수는 두 개의 반달 모양 데이터 포인트를 생성하는데 사용

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
-> 모듈에서 랜덤 포레스트 분류기와 여러 모델의 예측을 결합하는 앙상블 분류기 클래스 불러옴

from sklearn.linear_model import LogisticRegression
-> 모듈에서 로지스틱 회귀 분류기 만드는 클래스 불러옴
from sklearn.model_selection import train_test_split
-> 모듈에서 데이터셋을 학습 세트와 테스트 세트로 분할하는 함수 불러옴

from sklearn.svm import SVC
-> 모듈에서 svc 클래스 불러옴, 서포트 벡터머신 분류기를 만드는데 사용 

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
-> make_moons 함수 사용하여 500개의 샘플과 약간의 잡음을 가진 반달모양의 데이터셋 생성
x는 특성들 features 이고 y는 타겟 target 레이블이다

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
-> 데이터를 학습 세트와 테스트 세트로 분할한다

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
-> VotingClassifier를 생성하고, 세 개의 분류기를 앙상블로 사용한다
lr : 로지스틱 회귀 분류기, rf : 랜덤 포레스트 분류기, svc: 서포트 벡터머신 분류기

voting_clf.fit(X_train, y_train)
-> voting_clf 앙상블 분류기를 학습세트 X_train과 y_train에 맞춰 학습시킴

<결론>
이 코드는 세 가지 다른 머신러닝 모델 로지스틱 회귀, 랜덤 포레스트, 서포트 벡터머신을 결합한
앙상블 모델을 생성하여 학습 데이터를 기반으로 학습시킨다 앙상블 모델은 개별 모델들의 예측을
결합하여 더 강력하고 일반화된 예측 성능을 제공할 수 있게 된다 각 모델은 다른 알고리즘을 사용하여
학습하므로, 서로 다른 모델들의 강점들을 결합하여 더 나은 성능을 낼 수 있다
```
### 7.1 투표 기반 분류기 (5)
```
<테스트 세트에서 훈련된 각 분류기의 정확도 확인>

for name, clf in voting_clf.named_estimators_.items():
    print(name, "=", clf.score(X_test, y_test))
-> voting_clf 앙상블 모델의 각 개별 분류기의 이름과 해당 분류기의 테스트 세트에서의 정확도 출력
-> voting_clf.named_estimators.items() 통해 각 분류기의 이름과 객체 가져옴
clf.score(X_test, y_test) 호출하여 테스트 세트에서의 각 분류기의 정확도를 계산하고 출력 
```
```
<투표 기반 분류기의 predict() 메서드 호출하여 직접 투표 수행>
voting_clf.predict(X_test[:1])
-> voting_clf 앙상블 모델 사용하여 테스트 세트의 첫 번째 샘플에 대해 예측 수행
-> 출력은 예측된 클래스 레이블의 배열

array([1]) : 앙상블 모델이 테스트 세트의 첫 번째 샘플을 클래스 1로 예측

[clf.predict(X_test[:1]) for clf in voting_clf.estimators_]
-> 각 개별 분류기에서듸 테스트 세트의 첫 번째 샘플에 대해 예측을 수행
voting_clf.estimators_ 사용하여 각 분류기 객체를 가져옴
각 분류기의 predict() 메서드 호출하여 예측을 수행하고 결과를 리스트로 반환

[array([1]), array([1]), array([0])]는 각 분류기의 예측 결과로,
로지스틱 회귀, 랜덤 포레스트는 클래스 1을 예측, svc 클래스는 0을 예측

voting_clf.score(X_test, y_test)
-> voting_clf 앙상블 모델의 테스트 세트에서의 정확도를 계산하고 출력
score() 메소드는 주어진 데이터와 레이블에 대한 정확도를 반환
```


### 7.2 배깅과 페이스팅 (1)
```
- 배깅 bagging, bootstrap aggregating 의 줄임말
: 훈련 세트에서 중복을 허용하여 샘플링하는 방식
- 페이스팅 pasting
: 중복을 허용하지 않고 샘플링하는 방식
```
### 7.2 배깅과 페이스팅 (2)
```
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
-> 앙상블 학습의 한 방법인 배깅 분류기 제공, 여러 개의 학습기를 독립적으로 훈련시켜 그 예측을 결합
-> 결정 트리 분류기 제공, 개별 트리 학습기로 사용

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, n_jobs=-1, random_state=42)
-> BaggingClassifier 객체 생성, DecisionTreeClassifier() 배깅 앙상블에서 사용할 기본 학습기로
결정 트리 분류기를 지정, 500개의 결정 트리 학습기를 사용하여 앙상블을 구성,
각 기본 학습기는 100개의 샘플을 사용하여 학습, 이 샘플들은 원래 훈련 데이터셋에서 무작위로 선택
n_jobs=-1 모든 가용 CPU 코어를 사용하여 병렬로 학습 -> 모델의 학습 속도를 높여줌

bag_clf.fit(X_train, y_train)
-> BaggingClassifier 객체를 학습 세트에 맞춰서 학습시킴
fit 메소드는 배깅 앙상블 모델을 구성하는 500개의 결정트리 학습기를 각각 무작위로
선택된 100개의 샘플을 사용하여 학습시킴
```
### 7.2 배깅과 페이스팅 (3)
![image](https://github.com/simsoohyeon/Machine-Learning-Deep-Learning-Study/assets/127268889/83e895d7-3801-43ac-83f0-24ba3ce9040f)
```
단일 결정 트리의 결정 경계 vs 500개의 트리를 사용한 배깅 앙상블의 결정 경계 비교
- 왼쪽 그래프, 단일 경정 트리
결정 경계: 결정 트리가 학습한 데이터의 따라 매우 세밀하고 복잡한 경계 형성
트리가 과적합이 된 것으로 보임, 이는 데이터의 잡음과 세부적인 패턴을 과하게 학습한 결과
특징: 데이터의 포인트, 점과 사각형이 있는 곳에서 경계가 복잡하게 구불구불함
새로운 데이터에 대한 예측 성능이 낮을 수 있음

- 오른쪽 그래프, 배깅을 사용한 결정 트리
결정 경계: 여러 개의 결정 트리를 결합한 앙상블 모델의 경계
각 트리의 예측을 결합하여 더 부드럽고 일반화된 경계 형성
단일 트리에 비해 더 단순하고 덜 복잡한 경계 가지고 있음
특징: 결정 경계가 더 부드럽고 안정적이며 과적합을 줄임
다양한 트리의 예측을 결합하므로써 데이터의 잡음을 덜 민감하게 반응함
새로운 데이터에 대한 예측 성능이 더 높을 가능성이 있음

<결론>
- 단일 결정 트리는 데이터의 세부적인 패턴과 잡음을 모두 학습하여 과적합의 위험이 크다
- 배깅을 사용한 앙상블 모델은 여러 트리의 예측을 결합하여 더 일반화된 예측 제공
이는 모델이 과적합을 피하고 새로운 데이터에 대해 더 나은 성능을 발휘하도록 도움
앙상블 기법인 배깅을 통해 개별 모델의 단점을 보완하고 더 강력하고 안정적인 모델을 만듦
```
### 7.2 배깅과 페이스팅 (4)
```
< OOB 평가 설명 >
BaggingClassifier는 기본값을 중복으로 허용하여 (bootstrap=True) 훈련세트의 크기만큼
샘플을 선택함, 이는 평균적으로 각 예측기에 훈련 샘플의 약 63%정도만 샘플링됨
OOB (Out of Bag) 샘플은 선택되지 않은 나머지 37% 샘플을 말함

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            oob_score=True, n_jobs=-1, random_state=42)
-> 배깅 앙상블에서 사용할 기본 학습기로 결정 트리 분류기 지정,
n_estimators=500 500개의 결정 트리 학습기를 사용하여 앙상블을 구성,
oob_score=True로 지정하면 OOB 평가 활성화, 샘플을 사용하여 훈련이 끝난 후 자동으로 평가

bag_clf.fit(X_train, y_train)
-> BaggingClassifier 객체를 학습 세트 X_train과 y_train에 맞춰 학습
fit 메서드는 배깅 앙상블 모델을 구성하는 500개의 결정 트리 학습기를
각각 무작위로 선택된 샘플 사용하여 학습

bag_clf.oob_score_
-> OOB 샘플 사용하여 계산된 모델의 성능 점수 출력
obb_score_ 속성은 OOB 평가를 통해 계산된 모델의 정확도 나타냄


<요약>
이 코드는 배깅 앙상블을 사용하여 500개의 결정 트리 분류기를
학습시키고, OOB 평가를 통해 모델의 성능을 자동으로 평가합니다.
OOB 샘플은 모델의 성능을 검증하기 위해 사용되며, 이는 별도의
검증 세트를 사용하지 않고도 모델의 일반화 성능을 평가할 수 있는 방법입니다.
oob_score_ 속성은 훈련된 모델의 OOB 평가 정확도를 반환합니다.
이 예제에서는 0.896의 OOB 정확도를 얻었습니다.
```
### 7.2 배깅과 페이스팅 (5)
```
- OOB 평가 결과 확인
from sklearn.metrics import accuracy_score
-> 모듈에서 accuracy_score 함수 불러옴, 이 함수는 분류 모델의 정확도를 계산하는데 사용

y_pred = bag_clf.predict(X_test)
-> 학습된 배깅 분류기 bag_clf 를 사용하여 테스트 세트 X_test에 대한 예측 수행
predict 메소드는 입력 데이터에 대해 예측된 클래스 레이블 반환,
y_pred는 예측된 클래스 레이블  저장

accuracy_score(y_test, y_pred)
-> 테스트 세트 y_test와 예측된 클래스 레이블 y_pred를 사용하여 모델의 정확도 계산
accuracy_score 함수는 두 배열을 비교하여 정확도를 계산하고 그 결과를 반환
=> 모델의 정확도는 약 91.2%

- OOB 샘플에 대한 결정 함수의 값도 oob_decision_function_ 변수에서 확인
bag_clf.oob_decision_function_[:3] # 처음 3개의 샘플에 대한 확률
-> 학습된 배깅 분류기 bag_clf의 oob_decision_function 속성을 확인
이 속성은 OOB 샘플에 대한 결정 함수 값을 포함
[:3]은 처음 세 개의 샘플에 대해나 결정 함수의 값을 확인하는 것
=> 첫 번째 샘플 클래스 0에 속할 확률 32 클래스 1에 속할 확률 67
두 번째 샘플 클래스 0에 속할 확률 33 클래스 1에 속할 확률 66
세 번째 샘플 클래스 0에 속할 확률 100 클래스 1에 속할 확률 0

<요약>
이 코드는 배깅 분류기의 OOB 평가 결과와 테스트 세트에 대한 예측 결과를 확인
accurary_score 함수를 사용하여 테스트 세트의 정확도를 계산, 91%의 정확도 얻음
oob_decision_function_ 속성을 통해 OOB 샘플에 대한 각 클래스의 확률 값 확인
모델이 각 샘플에 대해 어떤 확률로 클래스를 예측했는지 보여줌 
```
### 7.3 랜덤 패치와 랜덤 서브스페이스
```
- 랜덤 패치 방식 random patches method
: 훈련 특성과 샘플을 모두 샘플링
모델의 다양성이 증가해 과적합을 줄일 수 있음
데이터의 하위 집합을 사용하기 때문에 학습 속도 증가가

- 랜덤 서브스페이스 방식 random subspaces method
: 훈련 샘플을 모두 사용 bootstrap = False 이고 max_samples=1.0 설정
특성을 샘플링 bootstrap_features=True로 설정, max_features는 1.0보다 작게 설정
모든 샘플을 다 사용하므로 데이터의 중요한 정보를 더 많이 포함하게 됨
특성 공간의 다양성을 높여 모델의 일반화 성능을 향상시킴 

이 방식은 특성 공간의 일부만의 사용하여 학습기를 훈련시키기 때문에,
특성 샘플링 모델은 더 다양한 예측기를 만들 수 있으며,
이는 편향을 늘리는 대신 분산을 낮추어 모델의 안정성을 높임

<결론>
랜덤 패치방식과 랜덤 서브스페이스 방식은 모두 데이터 샘플링을 통해 모델의 다양성을
높이고, 과적합을 줄이며 일반화 성능을 향상시키는데 유용함
랜덤 패치 방식은 특성과 샘플을 모두 샘플링하는 반면, 랜덤 서브스페이스 방식은
샘플은 모두 사용하고 특성만 샘플링한다
이러한 샘플링 방법은 앙상블 학습에서 모델의 성능을 향상시키는데 중요한 역
```
### 7.4 랜덤 포레스트 (1)

#### 랜덤 포레스트 분류기
```
from sklearn.ensemble import RandomForestClassifier
-> 모듈에서 RandomForestClassifier 클래스 불러옴, 랜덤포레스트 분류기 제공

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
-> RamdomForestClassifer 객체 생성, 500개의 결정 트리 사용하여 앙상블 구성,
각 결정 트리의 최대 리프 노드 수 16개로 제한

rnd_clf.fit(X_train, y_train)
-> 학습 세트 X_train과 y_train을 사용하여 랜덤 포레스트 분류기 학습
fit 메소드는 모델을 데이터에 맞춰 학습시킴

y_pred_rf = rnd_clf.predict(X_test)
-> 학습된 랜덤 포레스트 분류기를 사용하여 X_test에 대한 예측 수행
predict 메소드는 입력 데이터에 대해 예측된 클래스 레이블을 반환
y_pred_rf는 예측된 클래스 레이블 저장 
```
#### 배깅 분류기
```
from sklearn.ensemble import BaggingClassifier
-> 모듈에서 BaggingClassifier 클래스 불러옴, 배깅 분류기 제공

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500, n_jobs=-1, random_state=42)
-> BaggingClassifier 객체 생성, 배깅 앙상블에서 사용할 기본 학습기로 결정트리 분류기
sqrt는 각 결정 트리가 학습할 대 사용할 특성의 최대 수를 특성 수의 제곱근,
각 결정 트리의  최대 리프 노드 수를 16으로 제한, 500개의 결정 트리 학습시 사용
```
#### 요약
```
첫 번째는 랜덤 포레스트 분류기를 생성하고 학습시키고 테스트 세트에 대한 예측 수행
두 번째는 배깅 분류기 생성, 배깅 분류기는 결정 트리 분류기를 사용하여
앙상블을 구성하고, 여러 학습기를 병렬로 학습시키기 위해 모든 CPU 코어를 사용

두 모델 모두 앙상블 기법을 사용하여 더 강력하고 일반화된 모델을 만드는데 도움 줌
랜덤 포레스트는 여러 무작위 결정 트리를 학습시키고 배깅은 여러 학습기의 예측을 결
```










