from django import forms


TYPE =(
    ("1", "Small Object"),
    ("2", "Large Object"),
)


class objectCountingForm(forms.Form):
    image = forms.ImageField()
    type = forms.ChoiceField(choices=TYPE)
