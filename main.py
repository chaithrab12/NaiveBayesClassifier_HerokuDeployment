import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from scipy.sparse import hstack

app = Flask(__name__)
model = pickle.load(open('donorsChoose.pkl', 'rb'))
essayVect = pickle.load(open('essayTransform.pkl', 'rb'))
priceVect = pickle.load(open('price.pkl', 'rb'))
cleanCategoriesVect = pickle.load(open('cleanCategories.pkl', 'rb'))
cleanSubcategoriesVect = pickle.load(open('cleanSubcategories.pkl', 'rb'))
previouslyPostedProjectsCountVect = pickle.load(open('countOfPreviousProjectByteacher.pkl', 'rb'))
projectGradeVect = pickle.load(open('projectGradeCategory.pkl', 'rb'))
teacherPrefixVect = pickle.load(open('teacherPrefix.pkl', 'rb'))
schoolStateVect = pickle.load(open('schoolState.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    schoolState = request.form['school_state']
    schoolStateTransform = schoolStateVect.transform([schoolState])

    teacherPrefix = request.form['teacher_prefix']
    teacherPrefixTransform = teacherPrefixVect.transform([teacherPrefix])

    projectGradeCategory = request.form['project_grade_category']
    projectGradeCategoryTransform = projectGradeVect.transform([projectGradeCategory])

    previouslyPostedProjectsCount = request.form['teacher_number_of_previously_posted_projects']
    previouslyPostedProjectsCountTransform = previouslyPostedProjectsCountVect.transform(np.array(previouslyPostedProjectsCount).reshape(1,-1).reshape(-1,1))

    cleanCategories = request.form['clean_categories']
    cleanCategoriesTransform = cleanCategoriesVect.transform([cleanCategories])

    cleanSubcategories = request.form['clean_subcategories']
    cleanSubcategoriesTransform = cleanSubcategoriesVect.transform([cleanSubcategories])

    essay = request.form['essay']
    essayTransform = essayVect.transform([essay])

    price = request.form['price']
    priceTransform = priceVect.transform(np.array(price).reshape(1,-1).reshape(-1,1))

    print(essayTransform.shape,priceTransform.shape,schoolStateTransform.shape,
                             projectGradeCategoryTransform.shape,cleanCategoriesTransform.shape,cleanSubcategoriesTransform.shape, teacherPrefixTransform.shape,previouslyPostedProjectsCountTransform.shape)

    final_features = hstack((essayTransform,priceTransform,schoolStateTransform,
                             projectGradeCategoryTransform,cleanCategoriesTransform,cleanSubcategoriesTransform, teacherPrefixTransform,previouslyPostedProjectsCountTransform)).tocsr()
    print(final_features.shape)

    # final_features = [np.array(int_features)]

    prediction = model.predict(final_features)
    print(prediction)
    if prediction[0] == 1:
        print('Yes')
        return render_template('index.html', prediction_text='Project is eligible to approve '.format(prediction))
    else:
        print('No')
        return render_template('result.html', prediction_text='Project is not eligible to approve '.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
