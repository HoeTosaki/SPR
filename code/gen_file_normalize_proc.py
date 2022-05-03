import os

def normalize_file(out_dir='../tmp',out_file=None):
    if not os.path.exists(os.path.join(out_dir,out_file)):
        print('warning: file not exist',os.path.join(out_dir,out_file))
        return
    with open(os.path.join(out_dir,out_file),'r') as f1:
        with open(os.path.join(out_dir,out_file+'.tmp'),'w') as f2:
            while True:
                line = f1.readline()
                if line == "":
                    break
                lst = line.strip().split(',')
                assert len(lst) == 3
                if lst[0].startswith('tensor'):
                    src = int(lst[0].split('(')[1][:-1])
                else:
                    src = int(lst[0])
                if lst[1].startswith('tensor'):
                    dst = int(lst[1].split('(')[1][:-1])
                else:
                    dst = int(lst[1])
                dist = int(lst[2])
                f2.write(str(src)+','+str(dst)+','+str(dist)+'\n')
    os.system(r'rm -rf ' + os.path.join(out_dir, out_file))
    os.rename(os.path.join(out_dir, out_file)+'.tmp',os.path.join(out_dir, out_file))
if __name__ == '__main__':
    # for i in range(5):
    #     normalize_file(out_file='fb-landmarkinner~'+str(i))
    for i in range(5):
        normalize_file(out_file='tw-landmarkinner~'+str(i))
