import os
import json

def extract_value(path):

    newsTitles, newsContents, useTypes = [], [], []
    
    # 디렉토리가 실제로 존재하는지 확인
    if not os.path.exists(path):
        print(f"오류: '{path}' 경로를 찾을 수 없습니다.")
        return

    # 디렉토리 내의 파일 목록을 순회
    for filename in os.listdir(path):

        # .json 확인
        if filename.endswith(".json"):
            file_path = os.path.join(path, filename)
            
            try:
                # 파일 열기
                with open(file_path, 'r', encoding='utf-8') as f:
                    _data = json.load(f)

                    newsTitles.append(_data["sourceDataInfo"]["newsTitle"])

                    _content_list = [item.get("sentenceContent", "") for item in _data["sourceDataInfo"]["sentenceInfo"][:5]]
                    newsContents.append(" ".join(_content_list))
                    
                    useTypes.append(_data["sourceDataInfo"]["useType"])

                    
                    
            except json.JSONDecodeError:
                print(f"오류: {filename} 파일은 올바른 JSON 형식이 아닙니다.")
            except Exception as e:
                print(f"오류: {filename} 처리 중 문제 발생 - {e}")


    return newsTitles, newsContents, useTypes

if __name__ == "__main__":
    pass