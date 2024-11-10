from flask import render_template, redirect, url_for, flash
from flaskWebSite.models import INOUT, ClassificationImgs
from flaskWebSite.forms import UploadImgForm, SelectStuffForm, UploadImgFormForClassification
from flaskWebSite.utils import save_picture, predict_single_image, classes
from flaskWebSite import app, db


@app.route("/", methods=['GET','POST'])
def home():
    
    form = UploadImgForm()
    

    if form.validate_on_submit():

        return redirect(url_for('home'))
    
    return render_template("index.html", form=form, title = 'Ana Sayfa')

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    form = UploadImgFormForClassification()

    if form.validate_on_submit():
        picture_file = save_picture(form.img.data, 'static/classificationimages')
        image_path = "project_env/flaskWebSite/static/classificationimages/" + picture_file
        pred = predict_single_image(image_path)
        
        # Add the image and prediction to the database
        addable = ClassificationImgs(InputtedPic=picture_file, prediction=pred)
        db.session.add(addable)
        db.session.commit()
        
        # Pass the classes list to the template
        return render_template("class.html", form=form, title='Classification', pred=pred, image_id=addable.id, classes=classes)
    
    return render_template("class.html", form=form, title='Classification', classes=classes)


@app.route("/drawing", methods=['GET','POST'])
def drawing():
    
    
    return render_template("drawing.html", title = 'drawing')
