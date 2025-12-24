import argparse
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import json
import csv

import yaml


def load_yaml(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def build_runs(d_models, nheads, num_layers_list, lrs, wds, model_type='transformer'):
    """
    生成网格搜索参数组合
    
    Args:
        model_type: 'transformer', 'mamba', 或 'hybrid'
    """
    for d_model, nhead, num_layers, lr, wd in itertools.product(d_models, nheads, num_layers_list, lrs, wds):
        # Transformer类模型：需要检查d_model能否被nhead整除
        if model_type in ['transformer', 'hybrid']:
            if d_model % nhead != 0:
                continue
        
        yield {
            'd_model': int(d_model),
            'nhead': int(nhead),  # 对mamba模型，这个参数会被忽略
            'num_layers': int(num_layers),
            'dim_feedforward': int(d_model) * 4,
            'learning_rate': float(lr),
            'weight_decay': float(wd),
        }


def main():
    parser = argparse.ArgumentParser(description='Grid search for Transformer hyperparameters')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--script', type=str, default='train.py', help='Training entry script')
    parser.add_argument('--d_models', type=str, default='128,256,512')
    parser.add_argument('--nheads', type=str, default='8')  # Mamba模型会忽略此参数
    parser.add_argument('--num_layers', type=str, default='2,4,6,8')
    parser.add_argument('--lrs', type=str, default='1e-4,5e-5')
    parser.add_argument('--wds', type=str, default='1e-4,1e-3')
    parser.add_argument('--cuda', type=str, default=os.environ.get('CUDA_VISIBLE_DEVICES', ''), help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--gpu_ids', type=str, default=os.environ.get('GPU_IDS', ''), help='GPU_IDS (process local indices)')
    parser.add_argument('--dry_run', action='store_true', help='Only print planned runs without executing')
    parser.add_argument('--prefix', type=str, default='', help='Optional prefix added before task name')
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg_path = (root / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    script_path = (root / args.script) if not Path(args.script).is_absolute() else Path(args.script)

    if not cfg_path.exists():
        print(f'[ERROR] config file not found: {cfg_path}')
        sys.exit(1)
    if not script_path.exists():
        print(f'[ERROR] training script not found: {script_path}')
        sys.exit(1)

    def parse_list(s: str):
        return [x.strip() for x in s.split(',') if x.strip()]

    d_models = [int(x) for x in parse_list(args.d_models)]
    nheads = [int(x) for x in parse_list(args.nheads)]
    num_layers_list = [int(x) for x in parse_list(args.num_layers)]
    lrs = [float(x) for x in parse_list(args.lrs)]
    wds = [float(x) for x in parse_list(args.wds)]

    original_cfg = load_yaml(cfg_path)
    backup_path = cfg_path.with_suffix('.backup.yaml')
    dump_yaml(backup_path, original_cfg)
    print(f'[INFO] Backed up original config to {backup_path.name}')
    
    # 检查并显示模型类型
    model_type_full = original_cfg.get('model', {}).get('type', 'transformer').lower()
    supported_types = ['transformer', 'physics_transformer', 'enhanced_physics_transformer', 
                      'mamba_physics', 'hybrid_mamba_transformer', 'phys_tcn']
    
    if model_type_full not in supported_types:
        print(f'[WARN] Model type "{model_type_full}" may not be fully supported for grid search.')
        print(f'[INFO] Supported types: {", ".join(supported_types)}')
    else:
        print(f'[INFO] Grid search will update model.{model_type_full} configuration')
    
    # 根据模型类型确定build_runs的type参数
    if 'phys_tcn' in model_type_full:
        model_category = 'mamba'  # PhysTCN使用类似的参数结构（不需要nhead验证）
    elif 'mamba' in model_type_full and 'hybrid' not in model_type_full:
        model_category = 'mamba'
    elif 'hybrid' in model_type_full:
        model_category = 'hybrid'
    else:
        model_category = 'transformer'
    
    runs = list(build_runs(d_models, nheads, num_layers_list, lrs, wds, model_type=model_category))
    if not runs:
        print('[WARN] No valid run combinations after applying constraints.')
        return

    summaries = []
    results_root = Path('results') / 'grid_search'
    results_root.mkdir(parents=True, exist_ok=True)

    try:
        for i, r in enumerate(runs, 1):
            cfg = load_yaml(cfg_path)
            
            # 检测模型类型并选择对应的配置路径
            model_type = cfg.get('model', {}).get('type', 'transformer').lower()
            
            # 根据模型类型选择配置路径
            model_configs = {
                'transformer': 'transformer',
                'physics_transformer': 'physics_transformer',
                'enhanced_physics_transformer': 'enhanced_physics_transformer',
                'mamba_physics': 'mamba_physics',
                'hybrid_mamba_transformer': 'hybrid_mamba_transformer',
                'phys_tcn': 'phys_tcn',
            }
            
            config_key = model_configs.get(model_type)
            if not config_key:
                print(f'[ERROR] Model type "{model_type}" is not supported for grid search')
                break
            
            # 确保配置路径存在
            if config_key not in cfg.get('model', {}):
                print(f'[ERROR] Missing model.{config_key} section in config.yaml')
                break
            
            # 更新模型参数（根据模型类型更新不同的参数）
            m = cfg['model'][config_key]
            
            # Transformer类模型：更新nhead和dim_feedforward
            if model_type in ['transformer', 'physics_transformer', 'enhanced_physics_transformer']:
                m['d_model'] = r['d_model']
                m['nhead'] = r['nhead']
                m['num_layers'] = r['num_layers']
                m['dim_feedforward'] = r['dim_feedforward']
            
            # 纯Mamba模型：更新n_layers（不需要nhead和dim_feedforward）
            elif model_type == 'mamba_physics':
                m['d_model'] = r['d_model']
                m['n_layers'] = r['num_layers']
                # d_state, d_conv, expand保持配置文件中的默认值
            
            # 混合模型：更新nhead和层数分配
            elif model_type == 'hybrid_mamba_transformer':
                m['d_model'] = r['d_model']
                m['nhead'] = r['nhead']
                # 将总层数分配给mamba和transformer（简单策略：各一半）
                total_layers = r['num_layers']
                m['n_mamba'] = max(1, total_layers // 2)
                m['n_transformer'] = max(1, total_layers - m['n_mamba'])
            
            # PhysTCN模型：使用channels和num_layers（d_model映射为channels）
            elif model_type == 'phys_tcn':
                m['channels'] = r['d_model']  # d_model映射到channels
                m['num_layers'] = r['num_layers']
                # kernel_size, dropout等保持配置文件中的默认值
            
            # Keep dropout as-is in config (no sweep)

            # Training params
            cfg['training']['learning_rate'] = r['learning_rate']
            cfg['training']['weight_decay'] = r['weight_decay']
            
            # 生成task name：使用与train.py一致的格式，但添加lr和wd用于网格搜索区分
            # 设置前缀
            task_config = cfg.setdefault('task', {})
            if args.prefix:
                task_config['prefix'] = args.prefix.strip().replace(' ', '_')
            
            # 构建名称（格式：prefix_date_model_type_d{model}_nhead{head}_layer{layers}_feed{ff}_lr{lr}_wd{wd}_[suffix]）
            prefix = task_config.get('prefix', 'AAA')
            timestamp = datetime.now().strftime('%Y%m%d')
            
            # 名称部分（根据模型类型生成不同的名称）
            name_parts = [model_type]
            
            # Transformer类模型：包含d_model, nhead和dim_feedforward
            if model_type in ['transformer', 'physics_transformer', 'enhanced_physics_transformer']:
                name_parts.append(f"d{r['d_model']}")
                name_parts.append(f"nhead{r['nhead']}")
                name_parts.append(f"layer{r['num_layers']}")
                name_parts.append(f"feed{r['dim_feedforward']}")
            
            # 纯Mamba模型：包含d_model和层数
            elif model_type == 'mamba_physics':
                name_parts.append(f"d{r['d_model']}")
                name_parts.append(f"layer{r['num_layers']}")
                # d_state等参数从配置文件读取，添加到名称中
                name_parts.append(f"state{m.get('d_state', 16)}")
            
            # 混合模型：包含d_model, nhead和层数分配
            elif model_type == 'hybrid_mamba_transformer':
                name_parts.append(f"d{r['d_model']}")
                name_parts.append(f"mamba{m['n_mamba']}")
                name_parts.append(f"trans{m['n_transformer']}")
                name_parts.append(f"nhead{r['nhead']}")
            
            # PhysTCN模型：使用channels和num_layers
            elif model_type == 'phys_tcn':
                name_parts.append(f"ch{r['d_model']}")  # 使用ch表示channels
                name_parts.append(f"layer{r['num_layers']}")
                name_parts.append(f"k{m.get('kernel_size', 7)}")
            
            # 网格搜索特有：添加lr和wd以便区分不同超参数组合
            name_parts.append(f"lr{r['learning_rate']}")
            name_parts.append(f"wd{r['weight_decay']}")
            
            # 组合基础名称
            base_task_name = f"{prefix}_{timestamp}_{'_'.join(name_parts)}"
            
            # 添加后缀（如果存在）
            suffix_raw = task_config.get('suffix', '')
            if suffix_raw and str(suffix_raw).strip():
                run_name = f"{base_task_name}_{str(suffix_raw).strip()}"
            else:
                run_name = base_task_name
            
            # 设置task name
            task_config['name'] = run_name

            dump_yaml(cfg_path, cfg)
            print(f"[RUN {i}/{len(runs)}] {run_name}")

            if args.dry_run:
                continue

            env = os.environ.copy()
            if args.cuda:
                env['CUDA_VISIBLE_DEVICES'] = args.cuda
            if args.gpu_ids:
                env['GPU_IDS'] = args.gpu_ids

            start = time.time()
            proc = subprocess.run([sys.executable, str(script_path)], env=env)
            if proc.returncode != 0:
                print(f"[ERROR] Run failed: {run_name}")
                break
            duration_s = time.time() - start

            # Collect test metrics from results/test_result/<task_name>/metrics_*.json
            test_dir = Path('results') / 'test_result' / run_name
            mae = max_err = min_err = None
            if test_dir.exists():
                metric_files = sorted(test_dir.glob('metrics_*.json'))
                if metric_files:
                    with open(metric_files[-1], 'r', encoding='utf-8') as mf:
                        m = json.load(mf)
                        mae = m.get('mae', None)
                        max_err = m.get('max_err', None)
                        min_err = m.get('min_err', None)

            # 根据模型类型记录不同的参数
            summary = {
                'task_name': run_name,
                'model_type': model_type,
                'd_model': r['d_model'],
                'learning_rate': r['learning_rate'],
                'weight_decay': r['weight_decay'],
                'mae': mae,
                'max_err': max_err,
                'min_err': min_err,
                'duration_s': round(duration_s, 2)
            }
            
            # Transformer类模型：记录nhead和dim_feedforward
            if model_type in ['transformer', 'physics_transformer', 'enhanced_physics_transformer']:
                summary['nhead'] = r['nhead']
                summary['num_layers'] = r['num_layers']
                summary['dim_feedforward'] = r['dim_feedforward']
            
            # 纯Mamba模型：记录n_layers和d_state
            elif model_type == 'mamba_physics':
                summary['n_layers'] = r['num_layers']
                summary['d_state'] = m.get('d_state', 16)
                summary['d_conv'] = m.get('d_conv', 4)
                summary['expand'] = m.get('expand', 2)
            
            # 混合模型：记录mamba和transformer层数
            elif model_type == 'hybrid_mamba_transformer':
                summary['n_mamba'] = m['n_mamba']
                summary['n_transformer'] = m['n_transformer']
                summary['nhead'] = r['nhead']
                summary['d_state'] = m.get('d_state', 16)
            
            summaries.append(summary)
    finally:
        # Restore original config
        dump_yaml(cfg_path, original_cfg)
        print(f'[INFO] Restored original config.yaml')

    # Write summary files
    if summaries:
        tss = time.strftime('%Y%m%d_%H%M%S')
        csv_path = results_root / f'grid_summary_{tss}.csv'
        json_path = results_root / f'grid_summary_{tss}.json'
        
        # 动态生成fieldnames（根据第一个summary的keys）
        if summaries:
            fieldnames = list(summaries[0].keys())
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Wrote summary: {csv_path.name} & {json_path.name}")


if __name__ == '__main__':
    main()


