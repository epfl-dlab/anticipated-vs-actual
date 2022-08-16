lengths_dataset_path = 'anticipated_vs_actual/data/batches'

cramming_threshold_before = 115
cramming_threshold_after = 235

mobile_sources = ['<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>',
                 '<a href="http://twitter.com/download/android" rel="nofollow">Twitter for Android</a>',
                 '<a href="http://twitter.com/#!/download/ipad" rel="nofollow">Twitter for iPad</a>',
                 '<a href="https://mobile.twitter.com" rel="nofollow">Mobile Web (M2)</a>']

web_sources = ['<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>']

allowed_sources = web_sources + mobile_sources

allowed_languages = ['ar', 'nl','en', 'et', 'fr',\
                     'de','ht','hi', 'in', 'it', 'fa',\
                     'pl','pt', 'ru', 'es', 'sv', 'tl',\
                     'th', 'tr', 'ur']

lang_sorted = ['ht', 'ja', 'tl', 'ko', 'pt', 'th',\
               'et' ,'in', 'zh', 'ru', 'ar', 'fr',\
               'pl', 'es', 'it', 'en', 'nl', 'sv', \
               'de','tr', 'fa', 'ur', 'hi']

mapping_lang_codes = {'ja': 'Japanese',
                      'en': 'English',
                      'pt': 'Portuguese',
                      'es': 'Spanish',
                      'ar': 'Arabic',
                      'ko': 'Korean',
                      'in': 'Indonesian',
                      'tl': 'Tagalog',
                      'tr': 'Turkish',
                      'fr': 'French',
                      'th': 'Thai',
                      'ru': 'Russian',
                      'it': 'Italian',
                      'de': 'German',
                      'pl': 'Polish',
                      'hi': 'Hindi',
                      'fa': 'Persian',
                      'nl': 'Dutch',
                      'ht': 'Haitian Creole',
                      'et': 'Estonian',
                      'zh': 'Chinese',
                      'ur': 'Urdu',
                      'sv': 'Swedish'}
