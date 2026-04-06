from google import genai
from dotenv import load_dotenv
import os
import json
import time
import re
from pathlib import Path
from typing import Generator, Dict, Any, Optional

load_dotenv()

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-3.1-flash-lite-preview"

# Configurações
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / f"{model}_dados_sinteticos.jsonl"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
DELAY_BETWEEN_REQUESTS = 5  # Segundos entre requisições (ajuste conforme limite da API)


class SyntheticDataGenerator:
    """Gerador de base de dados sintética com checkpoint e tratamento robusto de JSONL."""
    
    def __init__(self, output_file: str = OUTPUT_FILE, checkpoint_file: str = CHECKPOINT_FILE):
        self.output_file = output_file
        self.checkpoint_file = checkpoint_file
        self.checkpoint = self._load_checkpoint()
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Carrega o checkpoint para retomar de onde parou."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    print(f"✓ Checkpoint carregado com {len(checkpoint.get('processed', []))} casos processados")
                    return checkpoint
            except Exception as e:
                print(f"⚠️ Erro ao carregar checkpoint: {e}. Iniciando do zero.")
        
        return {
            "processed": [],
            "errors": [],
            "total_processed": 0
        }
    
    def _save_checkpoint_success(self, prompt: str, context: str, total: int) -> None:
        """Salva sucesso de um caso no checkpoint."""
        case_id = f"{prompt}|{context}"
        if case_id not in self.checkpoint["processed"]:
            self.checkpoint["processed"].append(case_id)
        self.checkpoint["total_processed"] = total
        self._persist_checkpoint()
    
    def _save_checkpoint_error(self, prompt: str, context: str) -> None:
        """Marca um caso como erro no checkpoint."""
        case_id = f"{prompt}|{context}"
        if case_id not in self.checkpoint["errors"]:
            self.checkpoint["errors"].append(case_id)
        self._persist_checkpoint()
    
    def _persist_checkpoint(self) -> None:
        """Persiste o checkpoint em disco."""
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Erro ao salvar checkpoint: {e}")
    
    def _limpar_resposta_json(self, response_text: str) -> Optional[str]:
        """
        Remove marcadores de código markdown e limpa a resposta.
        Trata: ```json, ```jsonl, ```, etc.
        """
        # Remove blocos de código markdown
        response_text = re.sub(r'```(?:json|jsonl)?\n?', '', response_text)
        response_text = re.sub(r'```\n?', '', response_text)
        response_text = response_text.strip()
        
        return response_text
    
    def _validar_e_corrigir_jsonl(self, text: str) -> list[Dict[str, Any]]:
        """
        Valida e corrige possíveis erros menores em JSONL.
        Retorna lista de objetos JSON válidos.
        """
        lines = text.strip().split('\n')
        valid_objects = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
                valid_objects.append(obj)
            except json.JSONDecodeError as e:
                print(f"⚠️ Linha {i+1} inválida: {line[:100]}...")
                print(f"   Erro: {e}")
                # Tenta corrigir erros comuns
                corrected = self._tentar_corrigir_linha(line)
                if corrected:
                    try:
                        obj = json.loads(corrected)
                        valid_objects.append(obj)
                        print(f"   ✓ Corrigida com sucesso")
                    except json.JSONDecodeError:
                        print(f"   ✗ Impossível corrigir")
        
        return valid_objects
    
    def _tentar_corrigir_linha(self, line: str) -> Optional[str]:
        """Tenta corrigir erros comuns em linhas JSON."""
        try:
            # Remove aspas simples e substitui por duplas (tratamento ingênuo)
            if "'" in line and '"' not in line:
                line = line.replace("'", '"')
            
            # Tenta fazer parse
            json.loads(line)
            return line
        except json.JSONDecodeError:
            return None
    
    def gerar_prompt_com_contexto(self, path_prompt_base: str, path_contexto: str) -> Optional[str]:
        """
        Lê o prompt base e o contexto de arquivos locais e substitui 
        o marcador '---' pelo conteúdo do contexto.
        """
        try:
            with open(path_prompt_base, 'r', encoding='utf-8') as f:
                prompt_base = f.read()
            
            with open(path_contexto, 'r', encoding='utf-8') as f:
                contexto = f.read()
            
            if "---" in prompt_base:
                prompt_final = prompt_base.replace("---", contexto)
            else:
                print("⚠️ Aviso: Marcador '---' não encontrado no prompt base.")
                prompt_final = prompt_base
            
            return prompt_final
        except FileNotFoundError as e:
            print(f"❌ Erro: Arquivo não encontrado - {e.filename}")
            return None
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")
            return None
    
    def chamar_gemini(self, prompt: str) -> Optional[str]:
        """Faz requisição para a API Gemini e retorna a resposta."""
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"❌ Erro na requisição Gemini: {e}")
            return None
    
    def processar_resposta(self, response_text: str, context_path: str, prompt_path: str) -> list[Dict[str, Any]]:
        """
        Processa a resposta do Gemini, limpa e valida JSONL, adiciona contexto e prompt.
        Valida campos obrigatórios: instruct, response, context, prompt.
        """
        if not response_text:
            return []
        
        # Limpa marcadores de markdown
        limpa = self._limpar_resposta_json(response_text)
        
        # Valida e corrige JSONL
        objetos = self._validar_e_corrigir_jsonl(limpa)
        
        # Filtra objetos válidos e adiciona campos obrigatórios
        objetos_validos = []
        for obj in objetos:
            # Valida campos obrigatórios
            if "instruct" not in obj or not obj["instruct"]:
                print(f"   ⚠️ Campo 'instruct' ausente ou vazio")
                continue
            if "response" not in obj or not obj["response"]:
                print(f"   ⚠️ Campo 'response' ausente ou vazio")
                continue
            
            obj["context"] = context_path
            obj["prompt"] = prompt_path
            objetos_validos.append(obj)
        
        if len(objetos_validos) < len(objetos):
            print(f"   ⚠️ {len(objetos) - len(objetos_validos)} registros descartados por validação")
        
        return objetos_validos
    
    def salvar_batch(self, objetos: list[Dict[str, Any]]) -> None:
        """Salva um batch de objetos no arquivo JSONL."""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                for obj in objetos:
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            print(f"✓ {len(objetos)} registros salvos em {self.output_file}")
        except Exception as e:
            print(f"❌ Erro ao salvar batch: {e}")
    
    def listar_arquivos_conhecimento(self, base_dir: str = "base_conhecimento") -> Generator[tuple[str, str], None, None]:
        """
        Generator que itera sobre todos os arquivos .md na base de conhecimento.
        Yielda (path_relativo, path_absoluto).
        """
        base_path = Path(base_dir)
        
        if not base_path.exists():
            print(f"❌ Diretório {base_dir} não encontrado")
            return
        
        for md_file in sorted(base_path.rglob("*.md")):
            yield str(md_file)
    
    def listar_prompts(self, prompt_dir: str = "prompt") -> list[tuple[str, str]]:
        """Lista todos os prompts disponíveis."""
        prompt_path = Path(prompt_dir)
        
        if not prompt_path.exists():
            print(f"❌ Diretório {prompt_dir} não encontrado")
            return []
        
        prompts = []
        for md_file in sorted(prompt_path.glob("*.md")):
            prompts.append((str(md_file), md_file.name))
        
        return prompts
    
    def ja_processado(self, prompt_name: str, context_path: str) -> bool:
        """Verifica se este par prompt-contexto já foi processado com sucesso."""
        case_id = f"{prompt_name}|{context_path}"
        return case_id in self.checkpoint["processed"]
    
    def executar_geracao(self, delay: float = DELAY_BETWEEN_REQUESTS, 
                        prompt_dir: str = "prompt",
                        conhecimento_dir: str = "base_conhecimento") -> None:
        """
        Executa a geração completa da base de dados sintética.
        
        Args:
            delay: Tempo de espera entre requisições (em segundos)
            prompt_dir: Diretório com os prompts
            conhecimento_dir: Diretório com a base de conhecimento
        """
        prompts = self.listar_prompts(prompt_dir)
        
        if not prompts:
            print("❌ Nenhum prompt encontrado!")
            return
        
        total_processado = 0
        
        try:
            for prompt_path, prompt_name in prompts:
                for context_path in self.listar_arquivos_conhecimento(conhecimento_dir):
                    # Pula se já foi processado com sucesso
                    if self.ja_processado(prompt_name, context_path):
                        print(f"⏭️  Pulando (já processado): {prompt_name} + {context_path}")
                        continue
                    
                    print(f"\n📝 Processando: {prompt_name} + {context_path}")
                    
                    # Gera prompt com contexto
                    prompt_final = self.gerar_prompt_com_contexto(prompt_path, context_path)
                    if not prompt_final:
                        self._save_checkpoint_error(prompt_name, context_path)
                        continue
                    
                    # Chama API Gemini
                    print("🔄 Chamando Gemini...")
                    resposta = self.chamar_gemini(prompt_final)
                    if not resposta:
                        self._save_checkpoint_error(prompt_name, context_path)
                        print(f"❌ Erro na requisição para: {prompt_name} + {context_path}")
                        continue
                    
                    # Processa resposta com validação de campos
                    objetos = self.processar_resposta(resposta, context_path, prompt_name)
                    
                    if objetos:
                        self.salvar_batch(objetos)
                        total_processado += len(objetos)
                        self._save_checkpoint_success(prompt_name, context_path, total_processado)
                        print(f"✓ Caso processado com sucesso")
                    else:
                        self._save_checkpoint_error(prompt_name, context_path)
                        print(f"❌ Nenhum registro válido extraído (erro de validação)")
                    
                    # Respeita delay da API
                    print(f"⏱️  Aguardando {delay}s antes da próxima requisição...")
                    time.sleep(delay)
        
        except KeyboardInterrupt:
            print("\n\n⛔ Execução interrompida pelo usuário!")
            print(f"   Checkpoint salvo: {total_processado} registros processados")
        except Exception as e:
            print(f"\n❌ Erro durante execução: {e}")
        finally:
            processados = len(self.checkpoint["processed"])
            erros = len(self.checkpoint["errors"])
            print(f"\n✅ Execução finalizada!")
            print(f"   Total de registros: {total_processado}")
            print(f"   Casos com sucesso: {processados}")
            print(f"   Casos com erro: {erros}")
            print(f"   Arquivo de saída: {self.output_file}")
            print(f"   Checkpoint salvo em: {self.checkpoint_file}")


def main():
    """Função principal para executar a geração."""
    print("=" * 60)
    print("🚀 GERADOR DE BASE DE DADOS SINTÉTICA")
    print("=" * 60)
    
    generator = SyntheticDataGenerator()
    
    print("\n" + "=" * 60)
    print("Iniciando geração...")
    print("=" * 60)
    
    generator.executar_geracao(delay=DELAY_BETWEEN_REQUESTS)


if __name__ == "__main__":
    main()