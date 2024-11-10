from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField, BooleanField
from wtforms.validators import DataRequired
from flask_wtf.file import FileField, FileAllowed

class UploadImgForm(FlaskForm):
    img = FileField('Upload img (jpg,png,jpeg)', validators=[DataRequired(), FileAllowed(['jpg', 'png','jpeg'])])
    SelectionOne = SelectField('Type', choices=[('Select'), ('ONE'), ('TWO'), ('THREE'), ('FOUR'), ('FIVE')] ,validators=[DataRequired()] )
    SelectionTwo = SelectField('Type', choices=[('Select'), ('ONE'), ('TWO'), ('THREE')] ,validators=[DataRequired()] )
    submit = SubmitField('Upload')

class UploadImgFormForClassification(FlaskForm):
    img = FileField('Upload img (jpg,png,jpeg)', validators=[DataRequired(), FileAllowed(['jpg', 'png','jpeg'])])
    submit = SubmitField('Upload')

    
class SelectStuffForm(FlaskForm):

    choice_one = BooleanField()
    choice_two = BooleanField()

    submit = SubmitField('Select')
