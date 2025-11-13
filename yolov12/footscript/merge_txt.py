"""
文件目录结构如下：
rootdir /1/ 00000.txt、00001.txt、00002.txt 
    /2/ 00000.txt、00001.txt、00002.txt
    /3/ 00000.txt、00001.txt、00002.txt
outdir / 00000.txt、00001.txt、00002.txt
1、2、3里的所有同名txt文件（00000.txt、00001.txt、00002.txt）里面的内容，
合并到一个txt里面（3个00000.txt合并到一个00000.txt，3个00001.txt合并到1个00001.txt），合并后的文件路径在outdir下
"""
from pathlib import Path
import argparse

def parse():
  parser = argparse.ArgumentParser(description='Merge txt files in a directory into one file')
  parser.add_argument('--rootdir',type=Path,default=Path('./'),help='标签文件所处的文件夹')
  parser.add_argument('--outdir',type=Path,default=Path('./'),help='合并后文件存放的文件夹')
  return parser.parse_args()

def merge_txt(rootdir:Path,outdir:Path):
  txt_name = [] # 存储所有txt文件的名称
  subDirList = sorted([subDir for subDir in rootdir.iterdir() if subDir.is_dir()],key=lambda x: x.name) # 存储所有子文件夹的名称
  for subdir in subDirList:
    for txt in subdir.glob('*.txt'):
      txtName = txt.name
      if txtName not in txt_name: # 获取所有txt文件名称
        txt_name.append(txtName)
  for writename in txt_name:
    outfile = outdir / writename 
    with open(outfile,'w',encoding='utf-8') as wf:
      first = True
      # 将rootdir中所有同一名称的txt文件，合并到outdir下的一个txt文件中
      # 标签文件保存在了rootdir / subdir / txt_name中，不同subdir下有相同的txt_name
      
      for sub in subDirList:
 
        readfile = rootdir / sub / writename
        if readfile.exists():
          if not first:
            wf.write('\n')
          with open(readfile,'r',encoding='utf-8') as rf:
            for line in rf:
              wf.write(line)
          first = False
        else:
          continue
    print(f"{writename} merge done!")
  print(f"{len(txt_name)} txt files merge done")
    
def main():
  args = parse()
  rootdir = args.rootdir
  outdir = args.outdir
  if not rootdir.exists():
    raise FileNotFoundError(f"{rootdir} not exists")
  if not outdir.exists():
    outdir.mkdir(parents=True,exist_ok=True)   
  if outdir.resolve().is_relative_to(rootdir.resolve()):
    raise ValueError("outdir 不能在 rootdir 之内；请放到根外或并列目录。")
  
  merge_txt(rootdir,outdir)

if __name__ == '__main__':
  main()