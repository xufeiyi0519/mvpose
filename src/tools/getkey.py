

def get_key (dict, value):
    for k, v in dict.items():
        for i in range(len(v)):
            if v[i] == value:
                return k