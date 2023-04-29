import pandas as pd


def time_ann(path):
    r=pd.read_csv(path,sep=";")
    onset=[]
    offset=[]
    for i in range(len(r.index)):
        d=r.iloc[i,0]
        pos1=d.find('\t')
        pos2 = d.rfind('\t')
        onset.append(eval(d[pos1+1:pos2]))
        offset.append(eval(d[pos2 + 1:]))
    return onset,offset

def read_time():
    onset_question, offset_question=time_ann(path="F:/maryam_sh/dataset/stimuli/annotations/sound/sound_annotation_questions.tsv")
    onset_answer, offset_answer=time_ann(path="F:/maryam_sh/dataset/stimuli/annotations/sound/sound_annotation_sentences.tsv")
    # remove onset of question from onset of answer
    onset_question_2 = [int(x) for x in onset_question]
    offset_question_2 = [int(x) for x in offset_question]

    for i in onset_answer:
        if (int(i) in onset_question_2):
            onset_answer.remove(i)

    for i in onset_answer:
        if (i in onset_question):
            onset_answer.remove(i)

    for i in offset_answer:
        if (int(i) in offset_question_2):
            offset_answer.remove(i)

    for i in offset_answer:
        if (i in offset_question):
            offset_answer.remove(i)

    return onset_question, offset_question,onset_answer, offset_answer