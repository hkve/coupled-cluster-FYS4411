import re 

def reformat_tex(s):
    s = re.sub("\\\sum_{.*?}", "", s)
    s = re.sub(r"t_{(.*?),(.*?),(.*?),(.*?)}", r"t^{\1\2}_{\3\4}", s)
    s = re.sub(r"u_{(.*?),(.*?),(.*?),(.*?)}", r"u^{\1\2}_{\3\4}", s)
    s = re.sub(r"f_{(.*?),(.*?)}", r"f_{\1\2}", s)
    return s