from django import forms

class InputForm(forms.Form):
    stock = forms.IntegerField(widget=forms.NumberInput(
        attrs={
            'class' : "form-control",
            'id' : "stock" ,
            'name' : "stock", 
            'placeholder':"Enter your available stock here",
        }
    ))

    description = forms.CharField(max_length=100, widget=forms.Textarea(
        attrs={
            'class' : "form-control",
            'id' : "description" ,
            'name' : "description", 
            'placeholder':"Enter description of your product here",
        }
    ))