"""Synthetic Portuguese corpus (internal company policies) and a gold query set
for retrieval evaluation. All content is fabricated — no real company/policy."""

DOCUMENTS: list[dict] = [
    {"id": "pol_001", "title": "Home office", "text": (
        "Colaboradores podem trabalhar em regime de home office até 3 dias por semana, "
        "mediante acordo prévio com o gestor direto. É necessário garantir conexão de "
        "internet estável e disponibilidade durante o horário comercial."
    )},
    {"id": "pol_002", "title": "Reembolso de despesas", "text": (
        "Despesas de viagem a trabalho, como transporte, hospedagem e alimentação, podem "
        "ser reembolsadas mediante envio de nota fiscal em até 30 dias após a despesa. "
        "O reembolso é processado na folha de pagamento do mês seguinte."
    )},
    {"id": "pol_003", "title": "Política de férias", "text": (
        "Todo colaborador tem direito a 30 dias de férias por ano trabalhado, podendo ser "
        "divididas em até três períodos, sendo um deles de no mínimo 14 dias corridos. "
        "As férias devem ser solicitadas com 30 dias de antecedência."
    )},
    {"id": "pol_004", "title": "Segurança da informação", "text": (
        "É proibido compartilhar credenciais de acesso a sistemas internos com terceiros. "
        "Senhas devem ser trocadas a cada 90 dias e a autenticação de dois fatores é "
        "obrigatória para acesso a sistemas que contenham dados de clientes."
    )},
    {"id": "pol_005", "title": "Viagem corporativa", "text": (
        "Viagens a trabalho devem ser solicitadas com pelo menos 7 dias de antecedência "
        "através do sistema interno de viagens. Passagens aéreas e hospedagem são "
        "reservadas pela equipe de viagens corporativas, não pelo colaborador."
    )},
    {"id": "pol_006", "title": "Uso de equipamentos", "text": (
        "Notebooks e celulares fornecidos pela empresa são de uso profissional. Em caso de "
        "dano, perda ou roubo, o colaborador deve comunicar o time de TI em até 24 horas "
        "para bloqueio do equipamento."
    )},
    {"id": "pol_007", "title": "Licença maternidade e paternidade", "text": (
        "A licença maternidade tem duração de 180 dias, conforme programa Empresa Cidadã. "
        "A licença paternidade tem duração de 20 dias corridos, contados a partir do "
        "nascimento ou adoção da criança."
    )},
    {"id": "pol_008", "title": "Plano de saúde", "text": (
        "Todos os colaboradores CLT têm direito a plano de saúde a partir do primeiro dia "
        "de trabalho, com coparticipação de 20% em consultas e exames. Dependentes podem "
        "ser incluídos mediante custo adicional."
    )},
    {"id": "pol_009", "title": "Avaliação de desempenho", "text": (
        "A avaliação de desempenho ocorre semestralmente, envolvendo autoavaliação, "
        "avaliação do gestor e feedback 360 graus com pares. Os resultados influenciam "
        "diretamente promoções e ajustes salariais."
    )},
    {"id": "pol_010", "title": "Código de conduta", "text": (
        "Todo colaborador deve agir com integridade, respeito e transparência nas relações "
        "de trabalho. Situações de assédio ou discriminação devem ser reportadas ao canal "
        "de ética, que garante sigilo e não retaliação."
    )},
    {"id": "pol_011", "title": "Política de despesas com internet", "text": (
        "Colaboradores em regime de home office recebem um auxílio mensal fixo para custear "
        "internet e energia elétrica, pago junto com o salário, sem necessidade de "
        "comprovação de nota fiscal."
    )},
    {"id": "pol_012", "title": "Contratação de fornecedores", "text": (
        "Novos fornecedores devem passar por processo de due diligence antes da assinatura "
        "de contrato, incluindo verificação de certidões negativas e conformidade com a "
        "política anticorrupção da empresa."
    )},
    {"id": "pol_013", "title": "Segurança do trabalho", "text": (
        "Colaboradores que atuam presencialmente em áreas operacionais devem utilizar "
        "equipamentos de proteção individual (EPIs) obrigatórios, sob pena de afastamento "
        "imediato da atividade até regularização."
    )},
    {"id": "pol_014", "title": "Benefício de vale-alimentação", "text": (
        "O vale-alimentação é creditado no quinto dia útil de cada mês, no valor definido "
        "em convenção coletiva. O benefício não é cumulativo e expira ao final de cada mês."
    )},
    {"id": "pol_015", "title": "Política de desligamento", "text": (
        "Em caso de desligamento, o colaborador deve devolver todos os equipamentos da "
        "empresa em até 5 dias úteis. O acesso a sistemas internos é revogado no mesmo dia "
        "do desligamento."
    )},
]

# Gold query set: (query, relevant document ids, ranked by relevance)
QUERIES: list[dict] = [
    {"query": "posso trabalhar de casa quantos dias por semana?", "relevant_ids": ["pol_001"]},
    {
        "query": "como pedir reembolso de uma nota fiscal de hotel?",
        "relevant_ids": ["pol_002", "pol_005"],
    },
    {"query": "quantos dias de férias eu tenho direito?", "relevant_ids": ["pol_003"]},
    {"query": "de quanto em quanto tempo preciso trocar minha senha?", "relevant_ids": ["pol_004"]},
    {"query": "como faço para reservar uma viagem a trabalho?", "relevant_ids": ["pol_005"]},
    {"query": "perdi meu celular da empresa, o que eu faço?", "relevant_ids": ["pol_006"]},
    {"query": "quantos dias de licença paternidade eu tenho?", "relevant_ids": ["pol_007"]},
    {"query": "o plano de saúde cobre meus dependentes?", "relevant_ids": ["pol_008"]},
    {
        "query": "com que frequência acontece a avaliação de desempenho?",
        "relevant_ids": ["pol_009"],
    },
    {"query": "onde denunciar um caso de assédio no trabalho?", "relevant_ids": ["pol_010"]},
    {
        "query": "recebo ajuda de custo pra pagar internet trabalhando de casa?",
        "relevant_ids": ["pol_011", "pol_001"],
    },
    {
        "query": "preciso usar equipamento de proteção no trabalho presencial?",
        "relevant_ids": ["pol_013"],
    },
    {"query": "quando cai o vale-alimentação todo mês?", "relevant_ids": ["pol_014"]},
    {"query": "o que preciso devolver quando eu sair da empresa?", "relevant_ids": ["pol_015"]},
    {
        "query": "novos fornecedores passam por que tipo de verificação?",
        "relevant_ids": ["pol_012"],
    },
]
