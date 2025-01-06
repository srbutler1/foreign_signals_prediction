import re
from datetime import datetime, timezone, timedelta

def update_readme(readme_path='README.md'):
    with open(readme_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    if '## Last Updated' not in content:
        content += '\n\n## Last Updated\n<!-- LAST_UPDATED -->'
    
    utc_time = datetime.now(timezone.utc)
    central_time = utc_time - timedelta(hours=6)
    current_date = central_time.strftime('%Y-%m-%d %H:%M:%S CST')
    
    updated_content = re.sub(
        r'## Last Updated\n.*?(?=\n\n|$)', 
        f'## Last Updated\n{current_date}', 
        content,
        flags=re.DOTALL
    )
    
    with open(readme_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)

if __name__ == "__main__":
    update_readme()