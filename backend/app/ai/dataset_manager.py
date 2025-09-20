import os
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class BrazilianSentimentDatasets:
    """Gerenciador de datasets brasileiros para análise de sentimentos"""
    
    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}
        self.processed_datasets = {}
        
    def _download_dataset_from_url(self, url: str, filename: str) -> bool:
        """Download de dataset de URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filepath = self.data_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Dataset {filename} baixado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao baixar dataset {filename}: {e}")
            return False
    
    def create_sample_brazilian_dataset(self) -> pd.DataFrame:
        """Cria dataset de exemplo com dados brasileiros"""
        logger.info("Criando dataset de exemplo brasileiro...")
        
        # Dataset expandido com expressões brasileiras
        data = {
            'text': [
                # Muito Positivos
                "Cara, esse produto é sensacional demais! Recomendo muito!",
                "Mano, que experiência incrível! Superou todas minhas expectativas!",
                "Nossa, que maravilha! Melhor compra que já fiz na vida!",
                "Perfeito em todos os aspectos, vale muito a pena!",
                "Excelente qualidade, chegou super rápido! Top demais!",
                "Amo esse produto, uso todo dia! Fantástico mesmo!",
                "Show de bola! Funcionou perfeitamente, muito satisfeito!",
                "Que coisa linda! Design incrível e qualidade excepcional!",
                
                # Positivos
                "Gostei bastante, produto de boa qualidade",
                "Muito bom, recomendo sim! Vale a pena comprar",
                "Bacana, funcionou direitinho como esperado",
                "Legal, chegou no prazo e estava bem embalado",
                "Produto bom, custo benefício excelente",
                "Satisfeito com a compra, atendeu minhas necessidades",
                "Bonito e funcional, gostei do acabamento",
                "Bom produto, sem reclamações até agora",
                
                # Neutros
                "É um produto normal, nem bom nem ruim",
                "OK, funciona como descrito, nada demais",
                "Produto padrão, dentro do esperado mesmo",
                "Regular, adequado para o preço que paguei",
                "Comum, sem grandes surpresas positivas ou negativas",
                "Aceitável, mas poderia ser melhor em alguns aspectos",
                "Razoável, atende o básico que promete",
                "Normal, igual aos outros que já comprei",
                
                # Negativos  
                "Não gostei muito, qualidade deixa a desejar",
                "Produto fraco, não durou nem um mês",
                "Decepcionante, esperava muito mais pela descrição",
                "Não recomendo, muitos problemas logo no início",
                "Inferior ao que esperava, não vale o preço",
                "Chateado com a compra, produto veio com defeitos",
                "Insatisfeito, não funcionou como prometido",
                "Ruim, perdi dinheiro comprando isso",
                
                # Muito Negativos
                "Péssimo! Pior compra da minha vida, não funciona!",
                "Que lixo! Dinheiro jogado fora, produto horrível!",
                "Terrível experiência, produto chegou quebrado e mal feito!",
                "Odeio esse produto! Não comprem, é uma enganação!",
                "Horrível! Não dura nada e ainda é caro demais!",
                "Desastre total! Produto de quinta categoria, fujam!",
                "Lamentável qualidade, vergonhosa mesmo!",
                "Ridículo! Como vendem uma coisa dessas?",
                
                # Expressões com gírias e regionalismos
                "Massa demais, véi! Produto top das galáxias!",
                "Bagulho bom mesmo, sô! Recomendo pra galera!",
                "Tá louco, que produto ruim! Não presta não!",
                "Oxe, que coisa mais sem graça! Não gostei não!",
                "Uai, produto bão demais sô! Satisfeito eu!",
                "Bah, que legal guri! Produto tri bom!",
                "Poxa vida, que decepção! Esperava mais!",
                "Caramba, que experiência massa! Adorei!",
                
                # Com emojis e pontuação brasileira
                "Amei!!! ❤️ Produto maravilhoso mesmo! 😍",
                "Que raiva... produto ruim demais 😡💔",
                "Hmmm... não sei não... produto meio assim 🤔",
                "Nossa!!! Que surpresa boa! Adorei 🎉✨",
                "Poxa... que triste... produto ruim 😞",
                "Uau! Que incrível! Superou expectativas! 🤩",
                "Ah não... que decepção... 😔💸",
                "Show!!! Produto perfeito! 👏🏆",
                
                # Sarcasmo e ironia brasileira  
                "Que maravilha... produto que quebrou no primeiro dia 🙄",
                "Claro que funciona... quando quer né 😏",
                "Excelente qualidade... para o lixo mesmo 👎",
                "Perfeito... se você gosta de problemas 🤡",
                
                # Contextos específicos brasileiros
                "Produto chegou rapidinho pelos Correios! Muito bom!",
                "Demorou uma eternidade pra chegar, mas valeu a pena!",
                "Comprei no Black Friday, preço ótimo! Recomendo!",
                "Produto importado de qualidade, vale cada real!",
                "Made in China básico, mas funciona bem!",
                "Produto nacional de boa qualidade, orgulho!",
            ],
            
            'sentiment': [
                # Muito Positivos (8)
                'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive',
                # Positivos (8)  
                'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive',
                # Neutros (8)
                'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
                # Negativos (8)
                'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative',
                # Muito Negativos (8)
                'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative', 'negative',
                # Gírias (8)
                'positive', 'positive', 'negative', 'negative', 'positive', 'positive', 'negative', 'positive',
                # Com emojis (8)
                'positive', 'negative', 'neutral', 'positive', 'negative', 'positive', 'negative', 'positive',
                # Sarcasmo (4) - marcados como negativos pois são críticas sarcásticas
                'negative', 'negative', 'negative', 'negative',
                # Contextos brasileiros (6)
                'positive', 'positive', 'positive', 'positive', 'neutral', 'positive'
            ],
            
            'intensity': [
                # Muito Positivos (8) - intensidade alta
                0.9, 0.95, 0.9, 0.85, 0.9, 0.95, 0.85, 0.9,
                # Positivos (8) - intensidade média
                0.7, 0.75, 0.7, 0.65, 0.75, 0.7, 0.65, 0.7,
                # Neutros (8) - intensidade baixa
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                # Negativos (8) - intensidade média negativa
                0.7, 0.75, 0.7, 0.75, 0.7, 0.75, 0.7, 0.7,
                # Muito Negativos (8) - intensidade alta negativa
                0.95, 0.9, 0.95, 0.9, 0.85, 0.9, 0.85, 0.85,
                # Gírias (8)
                0.8, 0.8, 0.75, 0.7, 0.8, 0.8, 0.7, 0.85,
                # Com emojis (8)
                0.9, 0.8, 0.6, 0.85, 0.75, 0.9, 0.8, 0.85,
                # Sarcasmo (4) - intensidade alta (sarcasmo é intenso)
                0.8, 0.8, 0.85, 0.8,
                # Contextos brasileiros (6)
                0.75, 0.7, 0.8, 0.75, 0.6, 0.8
            ],
            
            'region': [
                # Distribuir por regiões do Brasil
                'SP', 'RJ', 'MG', 'RS', 'SP', 'RJ', 'PR', 'SC',  # Muito Positivos
                'BA', 'PE', 'CE', 'GO', 'DF', 'ES', 'MS', 'MT',  # Positivos
                'AM', 'PA', 'RO', 'AC', 'RR', 'AP', 'TO', 'MA',  # Neutros
                'PI', 'PB', 'RN', 'AL', 'SE', 'SP', 'RJ', 'MG',  # Negativos
                'RS', 'PR', 'SC', 'BA', 'PE', 'CE', 'GO', 'DF',  # Muito Negativos
                'BA', 'PE', 'MG', 'CE', 'RS', 'PR', 'SP', 'RJ',  # Gírias
                'SP', 'RJ', 'MG', 'RS', 'BA', 'PE', 'PR', 'SC',  # Emojis
                'SP', 'RJ', 'MG', 'RS',  # Sarcasmo
                'SP', 'RJ', 'RS', 'MG', 'PR', 'SC'  # Contextos brasileiros
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Salvar dataset
        dataset_path = self.data_dir / "brazilian_sentiment_sample.csv"
        df.to_csv(dataset_path, index=False, encoding='utf-8')
        logger.info(f"Dataset brasileiro criado: {dataset_path}")
        
        return df
    
    def load_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Carrega dataset específico"""
        if name in self.datasets:
            return self.datasets[name]
        
        dataset_path = self.data_dir / f"{name}.csv"
        if dataset_path.exists():
            try:
                df = pd.read_csv(dataset_path, encoding='utf-8')
                self.datasets[name] = df
                logger.info(f"Dataset {name} carregado: {len(df)} registros")
                return df
            except Exception as e:
                logger.error(f"Erro ao carregar dataset {name}: {e}")
                return None
        
        # Se não existe, tentar criar dataset de exemplo
        if name == "brazilian_sentiment_sample":
            return self.create_sample_brazilian_dataset()
        
        return None
    
    def preprocess_dataset(self, df: pd.DataFrame, text_column: str = 'text', label_column: str = 'sentiment') -> pd.DataFrame:
        """Preprocessa dataset para treinamento"""
        logger.info("Preprocessando dataset...")
        
        # Cópia do dataset
        processed_df = df.copy()
        
        # Limpeza básica de texto
        processed_df[text_column] = processed_df[text_column].astype(str)
        processed_df[text_column] = processed_df[text_column].str.strip()
        
        # Remover linhas vazias
        processed_df = processed_df[processed_df[text_column] != '']
        processed_df = processed_df[processed_df[text_column].notna()]
        
        # Normalizar labels
        if label_column in processed_df.columns:
            processed_df[label_column] = processed_df[label_column].str.lower().str.strip()
            
            # Mapear variações de labels
            label_mapping = {
                'positivo': 'positive', 'pos': 'positive', 'bom': 'positive',
                'negativo': 'negative', 'neg': 'negative', 'ruim': 'negative',  
                'neutro': 'neutral', 'neu': 'neutral', 'normal': 'neutral'
            }
            
            processed_df[label_column] = processed_df[label_column].replace(label_mapping)
            
            # Manter apenas labels válidos
            valid_labels = ['positive', 'negative', 'neutral']
            processed_df = processed_df[processed_df[label_column].isin(valid_labels)]
        
        # Estatísticas do dataset
        logger.info(f"Dataset preprocessado: {len(processed_df)} registros válidos")
        if label_column in processed_df.columns:
            label_counts = processed_df[label_column].value_counts()
            logger.info(f"Distribuição de labels: {dict(label_counts)}")
        
        return processed_df
    
    def split_dataset(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide dataset em treino e teste"""
        logger.info(f"Dividindo dataset: {test_size*100}% para teste")
        
        if 'sentiment' not in df.columns:
            # Split simples se não tem labels
            return train_test_split(df, test_size=test_size, random_state=random_state)
        
        # Split estratificado mantendo proporção de classes
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['sentiment']
        )
        
        logger.info(f"Dataset dividido: {len(train_df)} treino, {len(test_df)} teste")
        return train_df, test_df
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera estatísticas detalhadas do dataset"""
        stats = {
            'total_samples': len(df),
            'columns': list(df.columns),
            'text_stats': {},
            'label_distribution': {},
            'sample_texts': []
        }
        
        # Estatísticas de texto
        if 'text' in df.columns:
            texts = df['text'].astype(str)
            stats['text_stats'] = {
                'avg_length': float(texts.str.len().mean()),
                'min_length': int(texts.str.len().min()),
                'max_length': int(texts.str.len().max()),
                'avg_words': float(texts.str.split().str.len().mean()),
                'total_words': int(texts.str.split().str.len().sum())
            }
            
            # Amostras de textos
            stats['sample_texts'] = texts.sample(min(5, len(df))).tolist()
        
        # Distribuição de labels
        if 'sentiment' in df.columns:
            label_counts = df['sentiment'].value_counts()
            total = len(df)
            stats['label_distribution'] = {
                label: {
                    'count': int(count),
                    'percentage': float(count / total * 100)
                }
                for label, count in label_counts.items()
            }
        
        # Estatísticas regionais se disponível
        if 'region' in df.columns:
            region_counts = df['region'].value_counts()
            stats['regional_distribution'] = dict(region_counts.head(10))
        
        return stats
    
    def create_training_batch(self, df: pd.DataFrame, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Cria batches para treinamento"""
        batches = []
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            
            batch = {
                'texts': batch_df['text'].tolist(),
                'labels': batch_df['sentiment'].tolist() if 'sentiment' in batch_df.columns else None,
                'size': len(batch_df),
                'batch_id': i // batch_size
            }
            
            # Adicionar metadados se disponível
            if 'intensity' in batch_df.columns:
                batch['intensities'] = batch_df['intensity'].tolist()
            
            if 'region' in batch_df.columns:
                batch['regions'] = batch_df['region'].tolist()
            
            batches.append(batch)
        
        logger.info(f"Criados {len(batches)} batches de tamanho {batch_size}")
        return batches
    
    def export_dataset(self, df: pd.DataFrame, format: str = 'csv', filename: str = None) -> str:
        """Exporta dataset em diferentes formatos"""
        if filename is None:
            filename = f"processed_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        filepath = self.data_dir / f"{filename}.{format}"
        
        try:
            if format.lower() == 'csv':
                df.to_csv(filepath, index=False, encoding='utf-8')
            elif format.lower() == 'json':
                df.to_json(filepath, orient='records', force_ascii=False, indent=2)
            elif format.lower() == 'parquet':
                df.to_parquet(filepath, index=False)
            else:
                raise ValueError(f"Formato {format} não suportado")
            
            logger.info(f"Dataset exportado: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Erro ao exportar dataset: {e}")
            raise


class DatasetAnalyzer:
    """Analisador de qualidade e características de datasets"""
    
    def __init__(self):
        self.metrics = {}
    
    def analyze_text_quality(self, texts: List[str]) -> Dict[str, Any]:
        """Analisa qualidade dos textos no dataset"""
        if not texts:
            return {}
        
        # Estatísticas básicas
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        # Análise de caracteres especiais
        emoji_count = sum(1 for text in texts if any(ord(char) > 127 for char in text))
        
        # Análise de duplicatas
        unique_texts = set(texts)
        duplicate_rate = (len(texts) - len(unique_texts)) / len(texts) * 100
        
        # Análise de idioma (aproximada)
        portuguese_indicators = ['de', 'da', 'do', 'em', 'para', 'com', 'não', 'que', 'uma', 'um']
        portuguese_score = sum(
            sum(1 for word in portuguese_indicators if word in text.lower()) 
            for text in texts
        ) / len(texts)
        
        return {
            'total_texts': len(texts),
            'unique_texts': len(unique_texts),
            'duplicate_rate': round(duplicate_rate, 2),
            'length_stats': {
                'mean': round(np.mean(lengths), 2),
                'median': round(np.median(lengths), 2),
                'std': round(np.std(lengths), 2),
                'min': min(lengths),
                'max': max(lengths)
            },
            'word_stats': {
                'mean': round(np.mean(word_counts), 2),
                'median': round(np.median(word_counts), 2),
                'std': round(np.std(word_counts), 2),
                'min': min(word_counts),
                'max': max(word_counts)
            },
            'emoji_percentage': round(emoji_count / len(texts) * 100, 2),
            'portuguese_score': round(portuguese_score, 2),
            'quality_score': self._calculate_quality_score(lengths, word_counts, duplicate_rate)
        }
    
    def _calculate_quality_score(self, lengths: List[int], word_counts: List[int], duplicate_rate: float) -> float:
        """Calcula score de qualidade geral"""
        # Penalidades
        penalty = 0
        
        # Textos muito curtos
        very_short = sum(1 for length in lengths if length < 10) / len(lengths)
        penalty += very_short * 0.3
        
        # Textos muito longos (podem ser spam)
        very_long = sum(1 for length in lengths if length > 500) / len(lengths)
        penalty += very_long * 0.2
        
        # Taxa alta de duplicatas
        penalty += min(duplicate_rate / 100 * 0.4, 0.4)
        
        # Palavras por texto muito baixas
        avg_words = np.mean(word_counts)
        if avg_words < 3:
            penalty += 0.3
        
        # Score final (0-1, onde 1 é melhor)
        quality_score = max(0, 1 - penalty)
        return round(quality_score, 3)
    
    def analyze_label_balance(self, labels: List[str]) -> Dict[str, Any]:
        """Analisa balanceamento das classes"""
        if not labels:
            return {}
        
        label_counts = Counter(labels)
        total = len(labels)
        
        # Calcular distribuição
        distribution = {
            label: {
                'count': count,
                'percentage': round(count / total * 100, 2)
            }
            for label, count in label_counts.items()
        }
        
        # Calcular desbalanceamento
        percentages = [count / total for count in label_counts.values()]
        max_percentage = max(percentages)
        min_percentage = min(percentages)
        imbalance_ratio = max_percentage / min_percentage if min_percentage > 0 else float('inf')
        
        return {
            'total_labels': total,
            'unique_labels': len(label_counts),
            'distribution': distribution,
            'imbalance_ratio': round(imbalance_ratio, 2),
            'balance_score': round(1 / imbalance_ratio, 3) if imbalance_ratio != float('inf') else 0,
            'most_common': label_counts.most_common(1)[0] if label_counts else None,
            'least_common': label_counts.most_common()[-1] if label_counts else None
        }
    
    def generate_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera relatório completo do dataset"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        }
        
        # Análise de texto
        if 'text' in df.columns:
            texts = df['text'].dropna().astype(str).tolist()
            report['text_quality'] = self.analyze_text_quality(texts)
        
        # Análise de labels
        if 'sentiment' in df.columns:
            labels = df['sentiment'].dropna().tolist()
            report['label_analysis'] = self.analyze_label_balance(labels)
        
        # Recomendações
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas na análise"""
        recommendations = []
        
        # Recomendações de qualidade de texto
        if 'text_quality' in report:
            quality = report['text_quality']
            
            if quality.get('quality_score', 1) < 0.7:
                recommendations.append("Considere filtrar textos de baixa qualidade")
            
            if quality.get('duplicate_rate', 0) > 10:
                recommendations.append("Alta taxa de duplicatas - considere remover duplicadas")
            
            if quality.get('portuguese_score', 0) < 2:
                recommendations.append("Muitos textos podem não ser em português")
        
        # Recomendações de balanceamento
        if 'label_analysis' in report:
            balance = report['label_analysis']
            
            if balance.get('imbalance_ratio', 1) > 3:
                recommendations.append("Classes desbalanceadas - considere técnicas de balanceamento")
            
            if balance.get('unique_labels', 0) < 2:
                recommendations.append("Poucas classes - considere adicionar mais variabilidade")
        
        return recommendations
