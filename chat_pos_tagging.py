import nltk
from nltk.corpus import nps_chat

tagged_posts = nps_chat.tagged_posts(tagset='universal')
nr_posts = len(tagged_posts)

train_posts = tagged_posts[:(nr_posts*8)//10] #first 80% of the corpus
development_posts = tagged_posts[(nr_posts*8)//10:(nr_posts*9)//10] # 80-90% of the corpus

test_posts = tagged_posts[((nr_posts*9)//10):] # last 10% of the corpus
############################

dropout = 1e-4
train_posts = train_posts

#get most common tag for regex tagger
most_common_tag = nltk.FreqDist(tag for sent in train_posts for _, tag in sent).max()
#remove rare tokens
fd = nltk.FreqDist(word for sent in train_posts for word, _ in sent)
cfd = nltk.ConditionalFreqDist((word, tag) for sent in train_posts for word, tag in sent)
most_freq_words = [word for word, _ in fd.most_common(int(len(fd) * (1-dropout)))]
train_sents = [[(word, tag) if word in most_freq_words else ('UNK', tag) for word, tag in sent] for sent in train_posts]

patterns = [
    (r'(\:-*\)|\:-*\(|<3+|\:-*\*|\:-*\/|\:-*\||\:-*[bp]|;-*\)|[\:xX]-*D)', 'X'), # emojis
    (r'2moro|2nite|BRB|brb|BTW|btw|B4N|b4n|BCNU|bcnu|BFF|bff|CYA|cya|DBEYR|dbeyr|DILLIGAS|dilligas|FUD|fud|FWIW|fwiw|GR8|gr8|ILY|ily|IMHO|imho|IRL|irl|ISO|iso|J/K|j/k|L8R|l8r|LMAO|lmao|LOL|lol|LYLAS|lylas|MHOTY|mhoty|NIMBY|nimby|NP|np|NUB|nub|OIC|oic|OMG|omg|OT|ot|POV|pov|RBTL|rbtl|ROTFLMAO|rotflmao|RT|rt|THX|thx|SH|sh|SITD|sitd|SOL|sol|STBY|stby|SWAK|swak|TFH|tfh|RTM|rtm|TLC|tlc|TMI|tmi|TTYL|ttyl|TYVM|tyvm|VBG|vbg|WEG|weg|WTF|wtf|WYWH|wywh|XOXO|xoxo', 'X'), # chat abbreviations
    #(r'^the|a|some|most|every|no|which$', 'DET'), # determiners
    #(r'^on|of|at|with|by|into|under$', 'ADP'), # adpositions
    #(r'^he|their|her|its|my|I|us$', 'PRON'), # pronouns
    #(r'^at|on|out|over per|that|up|with$', 'PRT'), # particles
    #(r'^and|or|but|if|while|although$', 'CONJ'), # conjunctions
    (r'ly$', 'ADV'),   # adverbs
    (r'.*ing$', 'VERB'),   # gerunds
    (r'.*ed$', 'VERB'),    # simple past
    (r'.*es$', 'VERB'),    # 3rd singular present
    (r'.*ould$', 'VERB'),  # modals
    (r'.*\'s$', 'NOUN'),   # possessive nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'NUM'), # cardinal numbers
    (r'.*', most_common_tag)          # most common tag (default)
]

t0 = nltk.RegexpTagger(patterns)
t1 = nltk.UnigramTagger(train=train_sents, backoff=t0)
t2 = nltk.BigramTagger(train=train_sents, backoff=t1)
t3 = nltk.TrigramTagger(train=train_sents, backoff=t2)
print(t3.evaluate(development_posts)) #optimize parameters (e.g. dropout) on dev set

print(t3.evaluate(test_posts))
