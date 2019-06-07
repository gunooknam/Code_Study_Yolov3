'''
왼쪽 공백 지우기(lstrip)
오른쪽 공백 지우기(rstrip)
양쪽 공백 지우기(strip)
# 그래서 strip을 하는 것이다.
# 오른쪽 공백을 지운다.
a= " hi "
a.rstrip()
' hi'
# 양쪽 공백을 지우는 경우
a = " hi "
a.strip()
'hi'
s="Hello#World"
s.startswith('#')
False
s="#World"
s.startswith('#')
True
'''

def parse_model_config(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    # ['[net]', '# Testing', 'batch=1', 'subdivisions=1', '# Training', '# batch=64', '# subdivisions=2', 'width=416', 'height=416',
    #  'channels=3', 'momentum=0.9', 'decay=0.0005', 'angle=0', 'saturation = 1.5', 'exposure = 1.5', 'hue=.1', '', 'learning_rate=0.001',
    #  'burn_in=1000', 'max_batches = 500200', 'policy=steps', 'steps=400000,450000', 'scales=.1,.1', '', '# 0', '[convolutional]', 'batch_normalize=1',
    #  'filters=16', 'size=3', 'stride=1', 'pad=1', 'activation=leaky', '', '# 1', '[maxpool]', 'size=2', 'stride=2', '', '# 2', '[convolutional]',
    #  'batch_normalize=1', 'filters=32', 'size=3', 'stride=1', 'pad=1', 'activation=leaky', '', '# 3', '[maxpool]', 'size=2', 'stride=2', '', '# 4',
    # '[convolutional]', 'batch_normalize=1', 'filters=64', 'size=3', 'stride=1', 'pad=1', 'activation=leaky', '', '# 5', '[maxpool]', 'size=2', 'stride=2',
    # '', '# 6', '[convolutional]', 'batch_normalize=1', 'filters=128', 'size=3', 'stride=1', 'pad=1', 'activation=leaky', '', '# 7', '[maxpool]', 'size=2', 'stride=2',
    #  '', '# 8', '[convolutional]', 'batch_normalize=1', 'filters=256', 'size=3', 'stride=1', 'pad=1', 'activation=leaky', '', '# 9', '[maxpool]', 'size=2', 'stride=2',
    #  '', '# 10', '[convolutional]', 'batch_normalize=1', 'filters=512', 'size=3', 'stride=1', 'pad=1', 'activation=leaky', '', '# 11', '[maxpool]', 'size=2', 'stride=1',
    #  '', '# 12', '[convolutional]', 'batch_normalize=1', 'filters=1024', 'size=3', 'stride=1', 'pad=1', 'activation=leaky', '', '###########', '', '# 13', '[convolutional]',
    #  'batch_normalize=1', 'filters=256', 'size=1', 'stride=1', 'pad=1', 'activation=leaky', '', '# 14', '[convolutional]', 'batch_normalize=1', 'filters=512', 'size=3', 'stride=1',
    #  'pad=1', 'activation=leaky', '', '# 15', '[convolutional]', 'size=1', 'stride=1', 'pad=1', 'filters=255', 'activation=linear', '', '', '', '# 16', '[yolo]', 'mask = 3,4,5',
    #  'anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319', 'classes=80', 'num=6', 'jitter=.3', 'ignore_thresh = .7', 'truth_thresh = 1', 'random=1', '', '# 17', '[route]',
    #  'layers = -4', '', '# 18', '[convolutional]', 'batch_normalize=1', 'filters=128', 'size=1', 'stride=1', 'pad=1', 'activation=leaky', '', '# 19', '[upsample]', 'stride=2',
    # '', '# 20', '[route]', 'layers = -1, 8', '', '# 21', '[convolutional]', 'batch_normalize=1', 'filters=256', 'size=3', 'stride=1', 'pad=1', 'activation=leaky', '', '# 22',
    #  '[convolutional]', 'size=1', 'stride=1', 'pad=1', 'filters=255', 'activation=linear', '', '# 23', '[yolo]', 'mask = 1,2,3', 'anchors = 10,14,  23,27,  37,58,  81,82,  135,169,
    #   344,319', 'classes=80', 'num=6', 'jitter=.3', 'ignore_thresh = .7', 'truth_thresh = 1', 'random=1', '']
    # => cfg 파일에 있는 #은 주석이다.
    # lines에 x가
    lines = [x for x in lines if x and not x.startswith('#')] # 모든 공백을 지운다.
    lines = [x.rstrip().lstrip() for x in lines] # 좌우 공백을 다 지운다.
    module_defs = []
    for line in lines :
         if line.startswith('['):  # [ 으로 시작하는 것은
             module_defs.append({}) # 마지막 tuple
             module_defs[-1]['type'] = line[1:-1].rstrip() # type이라는 diction에 convolutional 이런 것을 저장
             if module_defs[-1]['type'] == 'convolutional':
                 module_defs[-1]['batch_normalize'] = 0 # batch norm은 일단 0으로 초기화
         else :
             key, value = line.split("=")
             value = value.strip()
             module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

'''
line="Line=spilt"
line.split("=")
['Line', 'spilt']
'''

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    # option 은 {} 와 같이 dict가 된다.
    # 그래서 그 후에 option['one']=1 이렇게 지정하면 { 'one': 1 }
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


# Testing
if __name__ == "__main__" :
    module_defs=parse_model_config("../config/yolov3-tiny.cfg")
    print(module_defs[0]['type']) # 그러고 나머지는 key : value 형식으로 저장
    print(module_defs[1]['type'])
    print(module_defs[2]['type'])
    print(module_defs[3]['type'])
    print(module_defs[4]['type'])
    '''
    net
    convolutional
    maxpool
    convolutional
    maxpool
    '''
