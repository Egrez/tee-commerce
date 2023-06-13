from django.shortcuts import render

from .forms import InputForm

from machinelearning.apps import MachinelearningConfig

import pandas as pd

def index(request):
	form = InputForm()

	if request.method == "POST":
		form = InputForm(data=request.POST)

		if form.is_valid():
			stock = form.cleaned_data['stock']
			description = form.cleaned_data['description']

			data = [[0 for i in range(271)]]
			columns = ['Items Sold', '2layer', '2xl', 'adult', 'aiden', 'american', 'animalprint', 'anime', 'asian', 'athletic', 'authentic', 'baby', 'badminton', 'basketball', 'beauty', 'beige', 'birthday', 'black', 'blackhorse', 'blouse', 'blue', 'bny', 'boutique', 'bowling', 'boy', 'breathable', 'buttondown', 'campinghiking', 'cartoon', 'casual', 'champion', 'checkedgeometric', 'chubby', 'collar', 'color', 'colors', 'combed', 'compressionsupport', 'cotton', 'cottonspandexblend', 'crewneck', 'croptop', 'crown', 'cycling', 'daily', 'denim', 'directfilm', 'disney', 'dress', 'drifit', 'durable', 'ecofriendly', 'elastic', 'emboss', 'embroidery', 'environmental', 'european', 'fashion', 'fashionable', 'female', 'fitness', 'floral', 'football', 'freeshipping', 'freesize', 'fruitloom', 'gift', 'girl', 'golf', 'graphic', 'gray', 'green', 'gym', 'hanes', 'hanford', 'hangdry', 'highneck', 'highquality', 'holiday', 'hooded', 'inklock', 'jersey', 'jogging', 'kid', 'knits', 'knitted', 'korean', 'large', 'lbust', 'licensed', 'llength', 'logo', 'longsleeve', 'loose', 'male', 'maroon', 'mbust', 'medium', 'mesh', 'microfiber', 'mlength', 'modern', 'moso', 'muscle', 'natural', 'nike', 'nonitch', 'notched', 'nylon', 'occasion', 'odorprotection', 'orange', 'organic', 'original', 'others', 'oversized', 'party', 'petrol', 'pink', 'plain', 'playing', 'plussize', 'polo', 'polyester', 'puma', 'purple', 'quarterturned', 'quickdry', 'recycled', 'red', 'regular', 'regularfit', 'reinforced', 'relaxedfit', 'retro', 'roughrider', 'roundneck', 'rugby', 'running', 'sbust', 'school', 'scoopneck', 'seamlessrib', 'semifit', 'shirtcollar', 'short', 'shortsleeve', 'silksatin', 'silkscreen', 'sizechart', 'sleeveless', 'slength', 'sletic', 'slim', 'small', 'soft', 'spandex', 'sport', 'sports', 'squareneck', 'streetwear', 'stretchable', 'stripe', 'stylisticmrlee', 'sublimation', 'summer', 'sunprotection', 'sweatabsorbent', 'sweater', 'swimming', 'tagless', 'tankscamisoles', 'tankshirt', 'tencel', 'tennis', 'training', 'travel', 'trendy', 'ultrasoft', 'unifit', 'uniform', 'unisex', 'urban', 'valentine', 'vintage', 'vinyl', 'vneck', 'washable', 'white', 'wicking', 'work', 'workout', 'xl', 'xlbust', 'xllength', 'xs', 'xxlbust', 'xxllength', 'yalex', 'yellow', 'yoga', 'yogastretch', 'youth', '1', 'active life', 'active-dryâ°', 'adidas', 'adventure bags', 'aiden sports', 'apple', 'bench', 'benoh', 'blue corner', 'bobson', 'burlington', 'cmge', 'coolair', 'daily grind clothing', 'decathlon', 'f.dyraa', 'fruit of the loom', 'gildan', 'goodlife', 'guitar', 'hghmnds clothing', 'huga underwears', 'huilishi', 'incerun', 'infinitee', 'inspi', 'jockey', 'jordan', 'kentucky', 'kingba', 'kinwoo.ph online shop', 'krave skin international', "levi's", 'lifeline', 'local brand', 'memo', 'monarchy katropa', 'monkey king', 'moose gear', 'ninety nine point nine boutique', 'no brand', 'oem', 'otaku', 'oxygen', 'penshoppe', 'regatta', 'rrj', 'skoop', 'sletic sports', 'softex', 'super flower', 'teetalk', 'tribal', 'under armour', 'walker clothing', 'world balance', 'yalex red label', 'zeneya', 'bulacan', 'cavite', 'cebu', 'ilocos norte', 'laguna', 'metro manila', 'overseas', 'pampanga', 'rizal']

			col_set = set(columns)

			test = pd.DataFrame(data=data, columns=columns)

			test['Items Sold'][0] = stock

			description = description.split(" ")

			for word in description:
				word = word.lower()

				if word in col_set:
					test[word] = 1

			test['Items Sold'] = MachinelearningConfig.items_scaler.transform(test['Items Sold'].values.reshape(-1, 1))

			predicted_price = round(MachinelearningConfig.model.predict(test)[0], 2)

			context = {
				'form' : form,
				'predicted_price' : predicted_price,
			}

		else:
			context = {
				'form' : form
			}
	
	else:
		context = {
			'form' : form,
		}
	

	return render(request, "index.html", context=context)
