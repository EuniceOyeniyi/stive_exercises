import random

def random_resp():
    random_list =[
        'Please trying writng something more descriptive',
        "Oh! it appears you wrote something I don't understand yet",
        "Do you mind trying to rephrase that?",
        "I'm terribly sorry, I don't quite catch that!",
        "I'm unable to give an answer to that yet, can I help with something else?"]

    list_count = len(random_list)
    random_item = random.randrange(list_count)
    return random_list[random_item]

