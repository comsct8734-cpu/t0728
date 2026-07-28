import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import io

# Streamlit 페이지 기본 설정
st.set_page_config(
    page_title="머신러닝 플레이그라운드 (수업용)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Streamlit UI refinement
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    .info-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 15px;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)

# 인터랙티브 캔버스 웹앱 HTML/JS/CSS 소스코드
HTML_CODE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>머신러닝 플레이그라운드</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Pretendard', sans-serif; user-select: none; }
        .tab-btn.active {
            border-bottom: 3px solid #3b82f6;
            color: #2563eb;
            font-weight: 700;
        }
        canvas {
            touch-action: none;
            cursor: crosshair;
        }
        /* Custom Tooltip Styling */
        .has-tooltip {
            position: relative;
            display: inline-flex;
            align-items: center;
            cursor: help;
        }
        .has-tooltip .tooltip-text {
            visibility: hidden;
            width: 220px;
            background-color: #1e293b;
            color: #fff;
            text-align: left;
            border-radius: 8px;
            padding: 8px 12px;
            position: absolute;
            z-index: 50;
            bottom: 125%;
            left: 50%;
            margin-left: -110px;
            opacity: 0;
            transition: opacity 0.2s, transform 0.2s;
            transform: translateY(4px);
            font-size: 0.75rem;
            font-weight: 400;
            line-height: 1.4;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            pointer-events: none;
        }
        .has-tooltip .tooltip-text::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #1e293b transparent transparent transparent;
        }
        .has-tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
            transform: translateY(0);
        }
    </style>
</head>
<body class="bg-slate-50 text-slate-800 p-2 md:p-4">
    <div class="max-w-7xl mx-auto bg-white rounded-2xl shadow-md border border-slate-200 overflow-hidden">
        
        <!-- Tab Navigation -->
        <div class="flex border-b border-slate-200 bg-slate-100/80 px-4 pt-3 gap-2 overflow-x-auto">
            <button onclick="switchTab('linear')" id="tab-linear" class="tab-btn active px-4 py-3 font-semibold text-slate-600 hover:text-blue-600 transition flex items-center gap-2 whitespace-nowrap">
                <i class="fa-solid fa-chart-line"></i> 1단계: 선형 회귀 (Linear)
            </button>
            <button onclick="switchTab('logistic')" id="tab-logistic" class="tab-btn px-4 py-3 font-semibold text-slate-600 hover:text-blue-600 transition flex items-center gap-2 whitespace-nowrap">
                <i class="fa-solid fa-square-poll-vertical"></i> 2단계: 로지스틱 회귀 (Logistic)
            </button>
            <button onclick="switchTab('knn')" id="tab-knn" class="tab-btn px-4 py-3 font-semibold text-slate-600 hover:text-blue-600 transition flex items-center gap-2 whitespace-nowrap">
                <i class="fa-solid fa-circle-nodes"></i> 3단계: K-최근접 이웃 (KNN)
            </button>
        </div>

        <!-- Notification Toast -->
        <div id="toast" class="hidden fixed bottom-5 right-5 bg-slate-800 text-white px-4 py-3 rounded-xl shadow-2xl z-50 text-sm flex items-center gap-2 transition-all">
            <i class="fa-solid fa-circle-info text-blue-400"></i> <span id="toast-msg">알림 메시지</span>
        </div>

        <div class="p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            <!-- Left Column: Canvas Area (7 Cols) -->
            <div class="lg:col-span-7 flex flex-col gap-4">
                <!-- Canvas Container -->
                <div class="relative bg-slate-900 rounded-2xl shadow-inner overflow-hidden border border-slate-800 aspect-square w-full max-h-[580px] flex items-center justify-center">
                    <canvas id="mlCanvas" class="w-full h-full block"></canvas>
                    
                    <!-- Prediction Floating Tooltip -->
                    <div id="predTooltip" class="absolute hidden bg-blue-600/90 backdrop-blur-sm text-white px-3 py-1.5 rounded-lg text-xs font-semibold shadow-lg pointer-events-none transition-all">
                        예측중...
                    </div>
                </div>

                <!-- Mode Indicator / Direct Instructions -->
                <div class="flex items-center justify-between text-xs md:text-sm bg-slate-100 p-3 rounded-xl border border-slate-200">
                    <div class="flex items-center gap-2 font-medium text-slate-700">
                        <i class="fa-solid fa-hand-pointer text-blue-500"></i>
                        <span id="interactionHint">캔버스를 좌클릭하여 데이터를 추가하세요. (우클릭: 점 삭제)</span>
                    </div>
                    <span id="pointCountBadge" class="bg-blue-100 text-blue-700 font-bold px-2.5 py-1 rounded-full text-xs">
                        데이터: 0개
                    </span>
                </div>
            </div>

            <!-- Right Column: Controls & Metrics (5 Cols) -->
            <div class="lg:col-span-5 flex flex-col gap-5">
                
                <!-- 1. Data Control Section -->
                <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <h3 class="font-bold text-slate-800 mb-3 flex items-center gap-2 text-sm md:text-base">
                        <i class="fa-solid fa-database text-blue-500"></i> 데이터 설정 & 업로드
                    </h3>
                    
                    <!-- CSV Upload -->
                    <div class="mb-3">
                        <label class="block text-xs font-semibold text-slate-600 mb-1">
                            <i class="fa-solid fa-file-csv text-green-600"></i> CSV 데이터 업로드
                            <span class="has-tooltip ml-1 text-slate-400"><i class="fa-solid fa-circle-question"></i>
                                <span class="tooltip-text">숫자 데이터가 포함된 CSV를 업로드하면 좌표계에 자동으로 정규화하여 시각화합니다.</span>
                            </span>
                        </label>
                        <input type="file" id="csvFileInput" accept=".csv" onchange="handleCSVUpload(event)" class="block w-full text-xs text-slate-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer">
                    </div>

                    <!-- Example Dataset Selection -->
                    <div class="mb-3">
                        <label class="block text-xs font-semibold text-slate-600 mb-1">탐구용 예시 데이터셋 선택</label>
                        <div class="grid grid-cols-2 gap-2" id="presetButtons">
                            <!-- Injected dynamically based on tab -->
                        </div>
                    </div>

                    <!-- Quick Random Point Generator -->
                    <div class="mb-3">
                        <label class="block text-xs font-semibold text-slate-600 mb-1">랜덤 데이터 빠른 추가</label>
                        <div class="flex items-center gap-2">
                            <button onclick="addRandomPoints(1)" class="flex-1 py-1.5 bg-white border border-slate-300 hover:bg-slate-100 rounded-lg text-xs font-semibold text-slate-700 transition">+1개</button>
                            <button onclick="addRandomPoints(5)" class="flex-1 py-1.5 bg-white border border-slate-300 hover:bg-slate-100 rounded-lg text-xs font-semibold text-slate-700 transition">+5개</button>
                            <button onclick="addRandomPoints(10)" class="flex-1 py-1.5 bg-white border border-slate-300 hover:bg-slate-100 rounded-lg text-xs font-semibold text-slate-700 transition">+10개</button>
                        </div>
                    </div>

                    <!-- Class selector for Classification Modes -->
                    <div id="classSelectorBox" class="hidden mt-3 pt-3 border-t border-slate-200">
                        <label class="block text-xs font-semibold text-slate-600 mb-1.5">추가할 클래스(범주) 선택</label>
                        <div class="flex gap-3">
                            <label class="flex-1 flex items-center justify-center gap-2 p-2 bg-red-50 border-2 border-red-400 rounded-lg cursor-pointer font-bold text-xs text-red-600">
                                <input type="radio" name="classSelect" value="0" checked onchange="selectedClass=0"> 클래스 0 (빨강)
                            </label>
                            <label class="flex-1 flex items-center justify-center gap-2 p-2 bg-blue-50 border-2 border-blue-400 rounded-lg cursor-pointer font-bold text-xs text-blue-600">
                                <input type="radio" name="classSelect" value="1" onchange="selectedClass=1"> 클래스 1 (파랑)
                            </label>
                        </div>
                    </div>

                    <!-- Clear / Action buttons -->
                    <div class="flex gap-2 mt-3">
                        <button onclick="clearData()" class="flex-1 py-2 bg-rose-50 hover:bg-rose-100 text-rose-600 border border-rose-200 rounded-xl font-bold text-xs transition flex items-center justify-center gap-1">
                            <i class="fa-solid fa-trash-can"></i> 전체 초기화
                        </button>
                        <button onclick="toggleMode()" id="modeToggleBtn" class="flex-1 py-2 bg-slate-800 hover:bg-slate-900 text-white rounded-xl font-bold text-xs transition flex items-center justify-center gap-1">
                            <i class="fa-solid fa-crosshairs"></i> <span id="modeToggleText">예측 모드 켜기</span>
                        </button>
                    </div>
                </div>

                <!-- 2. Hyperparameter & Model Learning -->
                <div class="bg-slate-50 border border-slate-200 rounded-xl p-4">
                    <h3 class="font-bold text-slate-800 mb-3 flex items-center gap-2 text-sm md:text-base">
                        <i class="fa-solid fa-sliders text-blue-500"></i> 하이퍼파라미터 & 학습
                    </h3>

                    <!-- Tab 1 & 2 Specific Controls -->
                    <div id="regressionControls" class="space-y-3">
                        <div>
                            <div class="flex justify-between text-xs font-semibold text-slate-600 mb-1">
                                <span class="has-tooltip">학습률 (Learning Rate, α)
                                    <span class="tooltip-text">가중치를 한 번에 얼마나 업데이트할지 결정합니다. 너무 크면 발산하고 너무 작으면 학습이 느립니다.</span>
                                </span>
                                <span id="lrVal" class="text-blue-600 font-bold">0.05</span>
                            </div>
                            <input type="range" id="lrSlider" min="0.001" max="0.3" step="0.005" value="0.05" oninput="document.getElementById('lrVal').innerText=this.value" class="w-full accent-blue-600 cursor-pointer">
                        </div>

                        <div>
                            <div class="flex justify-between text-xs font-semibold text-slate-600 mb-1">
                                <span class="has-tooltip">학습 횟수 (Epochs)
                                    <span class="tooltip-text">전체 데이터셋에 대해 경사하강법을 반복 실행할 횟수입니다.</span>
                                </span>
                                <span id="epochVal" class="text-blue-600 font-bold">50</span>
                            </div>
                            <input type="range" id="epochSlider" min="10" max="300" step="10" value="50" oninput="document.getElementById('epochVal').innerText=this.value" class="w-full accent-blue-600 cursor-pointer">
                        </div>

                        <div class="flex gap-2 pt-1">
                            <button onclick="startGradientDescent()" id="trainBtn" class="flex-1 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-bold text-xs transition shadow-md flex items-center justify-center gap-2">
                                <i class="fa-solid fa-play"></i> 경사하강법 학습 실행
                            </button>
                            <button onclick="computeOLS()" id="olsBtn" class="py-2.5 px-3 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 rounded-xl font-bold text-xs transition flex items-center gap-1">
                                <span class="has-tooltip">최소자승법(OLS)
                                    <span class="tooltip-text">수학 공식으로 단번에 해석적 최적해(정확한 직선)를 산출합니다.</span>
                                </span>
                            </button>
                        </div>
                    </div>

                    <!-- Tab 3 Specific KNN Controls -->
                    <div id="knnControls" class="hidden space-y-3">
                        <div>
                            <div class="flex justify-between text-xs font-semibold text-slate-600 mb-1">
                                <span class="has-tooltip">이웃 개수 (K Value)
                                    <span class="tooltip-text">새로운 데이터 판정 시 고려할 가장 가까운 이웃 데이터의 개수입니다.</span>
                                </span>
                                <span id="kVal" class="text-blue-600 font-bold">3</span>
                            </div>
                            <input type="range" id="kSlider" min="1" max="15" step="2" value="3" oninput="updateKValue(this.value)" class="w-full accent-blue-600 cursor-pointer">
                        </div>
                    </div>
                </div>

                <!-- 3. Metrics Dashboard -->
                <div class="bg-blue-950 text-white rounded-xl p-4 shadow-lg border border-blue-900">
                    <h3 class="font-bold mb-3 flex items-center gap-2 text-sm text-blue-300">
                        <i class="fa-solid fa-square-poll-round"></i> 실시간 모델 성능 & 파라미터
                    </h3>

                    <div id="metricsBox" class="grid grid-cols-2 gap-3 text-xs">
                        <!-- Metric items dynamic injection -->
                    </div>
                </div>

            </div>
        </div>
    </div>

    <script>
        // Global App State
        let currentTab = 'linear'; // 'linear', 'logistic', 'knn'
        let points = []; // [{x, y, class}] scaled inside [0, 1]
        let isPredictMode = false;
        let selectedClass = 0; // For logistic/knn
        let animationId = null;

        // Model Weights
        let weight = 0.0;
        let bias = 0.0;
        let w1 = 0.0, w2 = 0.0, b_log = 0.0; // Logistic regression
        let kKNN = 3;

        // Mouse hover predict state
        let hoverPos = null;

        // Canvas Setup
        const canvas = document.getElementById('mlCanvas');
        const ctx = canvas.getContext('2d');

        function resizeCanvas() {
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            draw();
        }
        window.addEventListener('resize', resizeCanvas);

        // Tab Switcher
        function switchTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(`tab-${tab}`).classList.add('active');

            // Toggle Controls
            const regControls = document.getElementById('regressionControls');
            const knnControls = document.getElementById('knnControls');
            const classBox = document.getElementById('classSelectorBox');
            const olsBtn = document.getElementById('olsBtn');

            if (tab === 'linear') {
                regControls.classList.remove('hidden');
                knnControls.classList.add('hidden');
                classBox.classList.add('hidden');
                olsBtn.classList.remove('hidden');
            } else if (tab === 'logistic') {
                regControls.classList.remove('hidden');
                knnControls.classList.add('hidden');
                classBox.classList.remove('hidden');
                olsBtn.classList.add('hidden');
            } else if (tab === 'knn') {
                regControls.classList.add('hidden');
                knnControls.classList.remove('hidden');
                classBox.classList.remove('hidden');
            }

            renderPresets();
            resetModelParameters();
            draw();
        }

        // Preset Data Generator
        function renderPresets() {
            const container = document.getElementById('presetButtons');
            container.innerHTML = '';

            let presets = [];
            if (currentTab === 'linear') {
                presets = [
                    { name: '📈 양의 상관관계', fn: loadPositiveLinear },
                    { name: '📉 음의 상관관계', fn: loadNegativeLinear },
                    { name: '🎲 낮은 상관관계', fn: loadLowCorrLinear },
                    { name: '⚠️ 이상치 포함 데이터', fn: loadOutlierLinear }
                ];
            } else {
                presets = [
                    { name: '🔴🔵 선형 분리 데이터', fn: loadSeparableData },
                    { name: '🟣 경계 중첩 데이터', fn: loadOverlappedData },
                    { name: '🎯 이상치 포함 데이터', fn: loadOutlierClassData }
                ];
            }

            presets.forEach(p => {
                const btn = document.createElement('button');
                btn.className = 'py-1.5 px-2 bg-white border border-slate-300 hover:bg-blue-50 hover:border-blue-300 rounded-lg text-xs font-semibold text-slate-700 transition truncate text-left';
                btn.innerText = p.name;
                btn.onclick = () => { p.fn(); showToast(`${p.name} 로드 완료`); };
                container.appendChild(btn);
            });
        }

        function showToast(msg) {
            const toast = document.getElementById('toast');
            document.getElementById('toast-msg').innerText = msg;
            toast.classList.remove('hidden');
            setTimeout(() => toast.classList.add('hidden'), 2500);
        }

        // Linear Presets
        function loadPositiveLinear() {
            points = [];
            for(let i=0; i<25; i++) {
                let x = 0.1 + Math.random() * 0.8;
                let y = 0.8 * x + 0.1 + (Math.random() - 0.5) * 0.15;
                points.push({x: clamp(x), y: clamp(y), class: 0});
            }
            computeOLS();
        }

        function loadNegativeLinear() {
            points = [];
            for(let i=0; i<25; i++) {
                let x = 0.1 + Math.random() * 0.8;
                let y = -0.75 * x + 0.85 + (Math.random() - 0.5) * 0.15;
                points.push({x: clamp(x), y: clamp(y), class: 0});
            }
            computeOLS();
        }

        function loadLowCorrLinear() {
            points = [];
            for(let i=0; i<30; i++) {
                let x = 0.1 + Math.random() * 0.8;
                let y = 0.2 + Math.random() * 0.6;
                points.push({x: clamp(x), y: clamp(y), class: 0});
            }
            computeOLS();
        }

        function loadOutlierLinear() {
            loadPositiveLinear();
            // Add 2 extreme outliers
            points.push({x: 0.2, y: 0.9, class: 0});
            points.push({x: 0.85, y: 0.15, class: 0});
            computeOLS();
        }

        // Classification Presets
        function loadSeparableData() {
            points = [];
            for(let i=0; i<15; i++) {
                let x = 0.15 + Math.random() * 0.35;
                let y = 0.15 + Math.random() * 0.35;
                points.push({x: clamp(x), y: clamp(y), class: 0});
            }
            for(let i=0; i<15; i++) {
                let x = 0.5 + Math.random() * 0.35;
                let y = 0.5 + Math.random() * 0.35;
                points.push({x: clamp(x), y: clamp(y), class: 1});
            }
            resetModelParameters();
            draw();
        }

        function loadOverlappedData() {
            points = [];
            for(let i=0; i<20; i++) {
                let x = 0.2 + Math.random() * 0.5;
                let y = 0.2 + Math.random() * 0.5;
                points.push({x: clamp(x), y: clamp(y), class: 0});
            }
            for(let i=0; i<20; i++) {
                let x = 0.35 + Math.random() * 0.5;
                let y = 0.35 + Math.random() * 0.5;
                points.push({x: clamp(x), y: clamp(y), class: 1});
            }
            resetModelParameters();
            draw();
        }

        function loadOutlierClassData() {
            loadSeparableData();
            points.push({x: 0.8, y: 0.8, class: 0}); // Outlier Red in Blue zone
            points.push({x: 0.2, y: 0.2, class: 1}); // Outlier Blue in Red zone
            draw();
        }

        function clamp(v) { return Math.max(0.02, Math.min(0.98, v)); }

        function addRandomPoints(count) {
            for(let i=0; i<count; i++) {
                let rx = 0.08 + Math.random() * 0.84;
                let ry = 0.08 + Math.random() * 0.84;
                let c = currentTab === 'linear' ? 0 : selectedClass;
                points.push({x: rx, y: ry, class: c});
            }
            if (currentTab === 'linear') computeOLS();
            else draw();
            showToast(`데이터 ${count}개 생성 완료`);
        }

        function clearData() {
            points = [];
            resetModelParameters();
            draw();
            showToast("데이터 초기화 완료");
        }

        function resetModelParameters() {
            weight = 0.0;
            bias = 0.5;
            w1 = 0.0; w2 = 0.0; b_log = 0.0;
        }

        function toggleMode() {
            isPredictMode = !isPredictMode;
            const btnText = document.getElementById('modeToggleText');
            const hint = document.getElementById('interactionHint');
            const tooltip = document.getElementById('predTooltip');

            if (isPredictMode) {
                btnText.innerText = "데이터 추가 모드로 전환";
                hint.innerText = "캔버스 위에 마우스를 올리면 실시간 예측을 수행합니다.";
            } else {
                btnText.innerText = "예측 모드 켜기";
                hint.innerText = "캔버스를 좌클릭하여 데이터를 추가하세요. (우클릭: 점 삭제)";
                tooltip.classList.add('hidden');
            }
            draw();
        }

        // Canvas Interactions (Click, Right-Click, Hover)
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = (e.clientX - rect.left) / rect.width;
            const mouseY = 1.0 - (e.clientY - rect.top) / rect.height; // Flip Y coordinate for math style

            if (e.button === 2) { // Right Click -> Delete Point
                e.preventDefault();
                deleteNearestPoint(mouseX, mouseY);
                return;
            }

            if (e.button === 0 && !isPredictMode) { // Left Click -> Add Point
                points.push({ x: mouseX, y: mouseY, class: currentTab === 'linear' ? 0 : selectedClass });
                if (currentTab === 'linear') computeOLS();
                else draw();
            }
        });

        canvas.addEventListener('contextmenu', e => e.preventDefault());

        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = (e.clientX - rect.left) / rect.width;
            const mouseY = 1.0 - (e.clientY - rect.top) / rect.height;

            hoverPos = { x: mouseX, y: mouseY, clientX: e.clientX, clientY: e.clientY };

            if (isPredictMode) {
                updatePredictionTooltip(mouseX, mouseY, e.clientX, e.clientY);
            }
            draw();
        });

        canvas.addEventListener('mouseleave', () => {
            hoverPos = null;
            document.getElementById('predTooltip').classList.add('hidden');
            draw();
        });

        function deleteNearestPoint(x, y) {
            if (points.length === 0) return;
            let minDist = Infinity;
            let targetIdx = -1;

            points.forEach((p, idx) => {
                let dist = Math.hypot(p.x - x, p.y - y);
                if (dist < minDist) {
                    minDist = dist;
                    targetIdx = idx;
                }
            });

            if (targetIdx !== -1 && minDist < 0.08) { // Delete radius tolerance
                points.splice(targetIdx, 1);
                showToast("데이터 점 1개 삭제 완료");
                if (currentTab === 'linear') computeOLS();
                else draw();
            }
        }

        // OLS (Ordinary Least Squares) for Linear Regression
        function computeOLS() {
            if (points.length < 2) {
                resetModelParameters();
                draw();
                return;
            }
            let n = points.length;
            let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
            points.forEach(p => {
                sumX += p.x;
                sumY += p.y;
                sumXY += p.x * p.y;
                sumXX += p.x * p.x;
            });

            let meanX = sumX / n;
            let meanY = sumY / n;

            let num = sumXY - n * meanX * meanY;
            let den = sumXX - n * meanX * meanX;

            if (Math.abs(den) < 1e-6) weight = 0;
            else weight = num / den;

            bias = meanY - weight * meanX;
            draw();
        }

        // Gradient Descent Animation
        function startGradientDescent() {
            if (points.length === 0) return;
            const lr = parseFloat(document.getElementById('lrSlider').value);
            const totalEpochs = parseInt(document.getElementById('epochSlider').value);
            let epoch = 0;

            if (currentTab === 'linear') {
                weight = 0.0; bias = 0.0;
            } else if (currentTab === 'logistic') {
                w1 = 0.0; w2 = 0.0; b_log = 0.0;
            }

            function step() {
                if (epoch >= totalEpochs) {
                    showToast("학습 완료!");
                    return;
                }

                let n = points.length;
                if (currentTab === 'linear') {
                    let dw = 0, db = 0;
                    points.forEach(p => {
                        let pred = weight * p.x + bias;
                        let err = pred - p.y;
                        dw += (2/n) * err * p.x;
                        db += (2/n) * err;
                    });
                    weight -= lr * dw;
                    bias -= lr * db;
                } else if (currentTab === 'logistic') {
                    let dw1 = 0, dw2 = 0, db = 0;
                    points.forEach(p => {
                        let z = w1 * p.x + w2 * p.y + b_log;
                        let a = 1 / (1 + Math.exp(-z));
                        let err = a - p.class;
                        dw1 += err * p.x;
                        dw2 += err * p.y;
                        db += err;
                    });
                    w1 -= (lr / n) * dw1;
                    w2 -= (lr / n) * dw2;
                    b_log -= (lr / n) * db;
                }

                epoch++;
                draw();
                animationId = requestAnimationFrame(step);
            }

            if (animationId) cancelAnimationFrame(animationId);
            step();
        }

        function updateKValue(val) {
            kKNN = parseInt(val);
            document.getElementById('kVal').innerText = kKNN;
            draw();
        }

        function updatePredictionTooltip(x, y, clientX, clientY) {
            const tooltip = document.getElementById('predTooltip');
            tooltip.classList.remove('hidden');
            const rect = canvas.getBoundingClientRect();
            tooltip.style.left = `${clientX - rect.left + 10}px`;
            tooltip.style.top = `${clientY - rect.top - 30}px`;

            if (currentTab === 'linear') {
                let predY = weight * x + bias;
                tooltip.innerText = `X: ${x.toFixed(2)} ➔ 예측 Y: ${predY.toFixed(2)}`;
            } else if (currentTab === 'logistic') {
                let z = w1 * x + w2 * y + b_log;
                let prob = 1 / (1 + Math.exp(-z));
                tooltip.innerText = `확률(클래스1): ${(prob * 100).toFixed(1)}%`;
            } else if (currentTab === 'knn') {
                let nearest = getKNN(x, y, kKNN);
                let class1Votes = nearest.filter(p => p.class === 1).length;
                let predClass = class1Votes > kKNN / 2 ? 1 : 0;
                tooltip.innerText = `KNN 예측: 클래스 ${predClass} (${class1Votes}/${kKNN} 표)`;
            }
        }

        function getKNN(x, y, k) {
            if (points.length === 0) return [];
            let sorted = [...points].map(p => ({
                ...p,
                dist: Math.hypot(p.x - x, p.y - y)
            })).sort((a, b) => a.dist - b.dist);
            return sorted.slice(0, Math.min(k, points.length));
        }

        // Render Master Loop
        function draw() {
            const w = canvas.width / window.devicePixelRatio;
            const h = canvas.height / window.devicePixelRatio;

            ctx.clearRect(0, 0, w, h);

            // Draw Background Grid Mesh for KNN decision boundaries or Logistic probabilities
            if (currentTab === 'knn' && points.length > 0) {
                drawKNNGrid(w, h);
            } else if (currentTab === 'logistic') {
                drawLogisticHeatmap(w, h);
            }

            // Draw Coordinate Grid Lines
            drawGridLines(w, h);

            // Draw Points & Model Specific Overlays
            if (currentTab === 'linear') {
                drawLinearRegression(w, h);
            } else if (currentTab === 'logistic') {
                drawLogisticBoundary(w, h);
            } else if (currentTab === 'knn') {
                drawKNNConnections(w, h);
            }

            drawPoints(w, h);
            updateMetricsDashboard();
            document.getElementById('pointCountBadge').innerText = `데이터: ${points.length}개`;
        }

        function drawGridLines(w, h) {
            ctx.strokeStyle = '#334155';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            for(let i=0.1; i<1.0; i+=0.1) {
                ctx.moveTo(i * w, 0); ctx.lineTo(i * w, h);
                ctx.moveTo(0, i * h); ctx.lineTo(w, i * h);
            }
            ctx.stroke();
        }

        function drawPoints(w, h) {
            points.forEach(p => {
                let cx = p.x * w;
                let cy = (1 - p.y) * h;

                ctx.beginPath();
                ctx.arc(cx, cy, 7, 0, Math.PI * 2);
                if (currentTab === 'linear') {
                    ctx.fillStyle = '#3b82f6';
                } else {
                    ctx.fillStyle = p.class === 0 ? '#ef4444' : '#3b82f6';
                }
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.stroke();
            });
        }

        function drawLinearRegression(w, h) {
            if (points.length === 0) return;

            // Draw Residuals (Error Lines)
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([3, 3]);
            points.forEach(p => {
                let predY = weight * p.x + bias;
                let cx = p.x * w;
                let cy = (1 - p.y) * h;
                let predCy = (1 - predY) * h;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(cx, predCy);
                ctx.stroke();
            });
            ctx.setLineDash([]);

            // Draw Fitted Line
            let y0 = weight * 0 + bias;
            let y1 = weight * 1 + bias;

            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 3.5;
            ctx.beginPath();
            ctx.moveTo(0, (1 - y0) * h);
            ctx.lineTo(w, (1 - y1) * h);
            ctx.stroke();
        }

        function drawLogisticHeatmap(w, h) {
            if (Math.abs(w1) < 1e-4 && Math.abs(w2) < 1e-4) return;
            const step = 8;
            for(let x=0; x<w; x+=step) {
                let nx = x / w;
                for(let y=0; y<h; y+=step) {
                    let ny = 1.0 - (y / h);
                    let z = w1 * nx + w2 * ny + b_log;
                    let prob = 1 / (1 + Math.exp(-z));
                    
                    ctx.fillStyle = prob > 0.5 ? `rgba(59, 130, 246, ${ (prob - 0.5) * 0.5 })` : `rgba(239, 68, 68, ${ (0.5 - prob) * 0.5 })`;
                    ctx.fillRect(x, y, step, step);
                }
            }
        }

        function drawLogisticBoundary(w, h) {
            if (Math.abs(w1) < 1e-4 && Math.abs(w2) < 1e-4) return;
            // Line equation: w1*x + w2*y + b = 0 => y = (-w1*x - b) / w2
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 3;
            ctx.beginPath();
            let x0 = 0, y0 = (-w1 * 0 - b_log) / w2;
            let x1 = 1, y1 = (-w1 * 1 - b_log) / w2;
            ctx.moveTo(x0 * w, (1 - y0) * h);
            ctx.lineTo(x1 * w, (1 - y1) * h);
            ctx.stroke();
        }

        function drawKNNGrid(w, h) {
            const step = 10;
            for(let x=0; x<w; x+=step) {
                let nx = x / w;
                for(let y=0; y<h; y+=step) {
                    let ny = 1.0 - (y / h);
                    let nearest = getKNN(nx, ny, kKNN);
                    let c1Votes = nearest.filter(p => p.class === 1).length;
                    
                    ctx.fillStyle = c1Votes > kKNN/2 ? 'rgba(59, 130, 246, 0.15)' : 'rgba(239, 68, 68, 0.15)';
                    ctx.fillRect(x, y, step, step);
                }
            }
        }

        function drawKNNConnections(w, h) {
            if (isPredictMode && hoverPos && points.length > 0) {
                let nearest = getKNN(hoverPos.x, hoverPos.y, kKNN);
                ctx.strokeStyle = '#f59e0b';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 4]);

                nearest.forEach(np => {
                    ctx.beginPath();
                    ctx.moveTo(hoverPos.x * w, (1 - hoverPos.y) * h);
                    ctx.lineTo(np.x * w, (1 - np.y) * h);
                    ctx.stroke();
                });
                ctx.setLineDash([]);
            }
        }

        // Metrics Calculation & UI Render
        function updateMetricsDashboard() {
            const container = document.getElementById('metricsBox');

            if (currentTab === 'linear') {
                let mse = 0, r2 = 0;
                if (points.length > 0) {
                    let sumErr = 0, sumTot = 0;
                    let meanY = points.reduce((acc, p) => acc + p.y, 0) / points.length;
                    points.forEach(p => {
                        let pred = weight * p.x + bias;
                        sumErr += Math.pow(p.y - pred, 2);
                        sumTot += Math.pow(p.y - meanY, 2);
                    });
                    mse = sumErr / points.length;
                    r2 = sumTot === 0 ? 1 : 1 - (sumErr / sumTot);
                }

                container.innerHTML = `
                    <div class="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                        <span class="text-slate-400 block text-[11px] has-tooltip">MSE (평균제곱오차)
                            <span class="tooltip-text">실제값과 예측값 차이의 제곱 평균입니다. 0에 가까울수록 모델 정확도가 높습니다.</span>
                        </span>
                        <span class="text-base font-bold text-emerald-400">${mse.toFixed(4)}</span>
                    </div>
                    <div class="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                        <span class="text-slate-400 block text-[11px] has-tooltip">R² Score (결정계수)
                            <span class="tooltip-text">회귀 모델이 데이터를 얼마나 설명하는지 나타냅니다. 1에 가까울수록 완벽한 모델입니다.</span>
                        </span>
                        <span class="text-base font-bold text-amber-400">${r2.toFixed(3)}</span>
                    </div>
                    <div class="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                        <span class="text-slate-400 block text-[11px] has-tooltip">기울기 (Weight w)
                            <span class="tooltip-text">X가 1단위 증가할 때 Y의 변화량 추정치입니다.</span>
                        </span>
                        <span class="text-base font-bold text-sky-400">${weight.toFixed(3)}</span>
                    </div>
                    <div class="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                        <span class="text-slate-400 block text-[11px] has-tooltip">절편 (Bias b)
                            <span class="tooltip-text">X가 0일 때 예측되는 Y의 기본값입니다.</span>
                        </span>
                        <span class="text-base font-bold text-purple-400">${bias.toFixed(3)}</span>
                    </div>
                `;
            } else if (currentTab === 'logistic') {
                let loss = 0, acc = 0;
                if (points.length > 0) {
                    let correct = 0;
                    points.forEach(p => {
                        let z = w1 * p.x + w2 * p.y + b_log;
                        let prob = Math.max(1e-5, Math.min(1 - 1e-5, 1 / (1 + Math.exp(-z))));
                        loss += -(p.class * Math.log(prob) + (1 - p.class) * Math.log(1 - prob));
                        let predClass = prob > 0.5 ? 1 : 0;
                        if (predClass === p.class) correct++;
                    });
                    loss /= points.length;
                    acc = (correct / points.length) * 100;
                }

                container.innerHTML = `
                    <div class="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                        <span class="text-slate-400 block text-[11px] has-tooltip">Log Loss (교차 엔트로피)
                            <span class="tooltip-text">분류 확률과 실제 클래스 차이를 측정하는 손실 함수입니다. 0에 가까울수록 좋습니다.</span>
                        </span>
                        <span class="text-base font-bold text-emerald-400">${loss.toFixed(4)}</span>
                    </div>
                    <div class="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                        <span class="text-slate-400 block text-[11px] has-tooltip">정확도 (Accuracy)
                            <span class="tooltip-text">전체 데이터 중 맞게 분류한 데이터의 비율입니다.</span>
                        </span>
                        <span class="text-base font-bold text-amber-400">${acc.toFixed(1)}%</span>
                    </div>
                `;
            } else if (currentTab === 'knn') {
                let looAcc = 0;
                if (points.length > 1) {
                    let correct = 0;
                    points.forEach((p, idx) => {
                        let others = points.filter((_, i) => i !== idx);
                        let sorted = others.map(op => ({...op, dist: Math.hypot(op.x - p.x, op.y - p.y)}))
                                           .sort((a,b) => a.dist - b.dist);
                        let topK = sorted.slice(0, Math.min(kKNN, others.length));
                        let c1 = topK.filter(k => k.class === 1).length;
                        let pred = c1 > topK.length / 2 ? 1 : 0;
                        if (pred === p.class) correct++;
                    });
                    looAcc = (correct / points.length) * 100;
                }

                container.innerHTML = `
                    <div class="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                        <span class="text-slate-400 block text-[11px] has-tooltip">K 값 (Neighbor Count)
                            <span class="tooltip-text">새 점 판정 시 비교할 가장 가까운 이웃 데이터 수입니다.</span>
                        </span>
                        <span class="text-base font-bold text-sky-400">${kKNN}</span>
                    </div>
                    <div class="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                        <span class="text-slate-400 block text-[11px] has-tooltip">LOO 교차검증 정확도
                            <span class="tooltip-text">Leave-One-Out 검증으로 측정한 KNN 분류 예측 정확도입니다.</span>
                        </span>
                        <span class="text-base font-bold text-amber-400">${looAcc.toFixed(1)}%</span>
                    </div>
                `;
            }
        }

        // CSV File Reader Parser & Min-Max Scaler
        function handleCSVUpload(e) {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(evt) {
                const text = evt.target.result;
                const lines = text.split('\\n').map(l => l.trim()).filter(l => l.length > 0);
                if (lines.length < 2) {
                    alert("유효한 CSV 파일 데이터가 부족합니다.");
                    return;
                }

                let parsed = [];
                for(let i=1; i<lines.length; i++) {
                    let parts = lines[i].split(',').map(Number);
                    if (parts.length >= 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
                        parsed.push({
                            xRaw: parts[0],
                            yRaw: parts[1],
                            cRaw: parts[2] !== undefined && !isNaN(parts[2]) ? parts[2] : 0
                        });
                    }
                }

                if (parsed.length === 0) {
                    alert("숫자 컬럼 데이터 추출 실패");
                    return;
                }

                // Auto Normalization (Min-Max Scaling)
                let minX = Math.min(...parsed.map(p => p.xRaw));
                let maxX = Math.max(...parsed.map(p => p.xRaw));
                let minY = Math.min(...parsed.map(p => p.yRaw));
                let maxY = Math.max(...parsed.map(p => p.yRaw));

                let rangeX = maxX - minX || 1;
                let rangeY = maxY - minY || 1;

                points = parsed.map(p => ({
                    x: clamp(0.1 + ((p.xRaw - minX) / rangeX) * 0.8),
                    y: clamp(0.1 + ((p.yRaw - minY) / rangeY) * 0.8),
                    class: p.cRaw > 0 ? 1 : 0
                }));

                showToast(`CSV 데이터 ${points.length}개 정규화 변환 완료!`);
                if (currentTab === 'linear') computeOLS();
                else draw();
            };
            reader.readAsText(file);
        }

        // Initialization on Load
        window.onload = function() {
            resizeCanvas();
            renderPresets();
            loadPositiveLinear();
        };
    </script>
</body>
</html>
"""

# Title and App Header
st.title("🤖 고등 머신러닝 플레이그라운드")
st.markdown("**개념을 직접 클릭하고 조작하며 이해하는 머신러닝 탐구 웹앱 (수업용)**")

# Sidebar Setup
with st.sidebar:
    st.header("📌 수업 가이드 & 도구")
    
    st.info("💡 **수업 활용 팁**\n- 캔버스 클릭으로 데이터점을 직접 추가해보세요.\n- **우클릭**으로 특정 점만 삭제할 수 있습니다.\n- 용어 옆 **물음표(?)**에 마우스를 대면 개념 설명이 나옵니다.")
    
    st.divider()
    
    # CSV Helper Generator for Class Test
    st.subheader("📥 수업용 실습 CSV 다운로드")
    st.caption("학생들이 직접 다운로드받아 캔버스에 업로드할 수 있는 예제 파일입니다.")
    
    # 1. Study time & Score CSV
    df_linear = pd.DataFrame({
        "StudyHours_X": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "ExamScore_Y": [52, 58, 63, 68, 74, 80, 85, 89, 93, 98]
    })
    csv_bytes1 = df_linear.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 [선형회귀] 공부시간-성적.csv",
        data=csv_bytes1,
        file_name="study_hours_vs_score.csv",
        mime="text/csv"
    )
    
    # 2. Classification CSV
    df_class = pd.DataFrame({
        "Feature1_X1": [1.2, 1.8, 2.3, 2.9, 6.1, 6.8, 7.5, 8.2],
        "Feature2_X2": [1.5, 2.1, 1.9, 2.7, 7.2, 6.9, 8.1, 7.8],
        "Class_Label": [0, 0, 0, 0, 1, 1, 1, 1]
    })
    csv_bytes2 = df_class.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 [분류] 2차원_범주형_데이터.csv",
        data=csv_bytes2,
        file_name="classification_sample.csv",
        mime="text/csv"
    )

# Render main interactive canvas view via HTML component
st.components.v1.html(HTML_CODE, height=920, scrolling=True)

# Deployment Guide Expander
with st.expander("🚀 깃허브(GitHub) & 스트림릿 커뮤니티 클라우드 배포 방법 안내", expanded=False):
    st.markdown("""
    ### 1단계: 깃허브(GitHub) 저장소(Repository) 생성
    1. [GitHub](https://github.com) 로그인 후 **New Repository** 클릭
    2. 저장소 이름(Repository Name) 입력 (예: `ml-playground`)
    3. **Public** 선택 후 `Create repository` 클릭

    ### 2단계: 코드 업로드
    저장소에 다음 두 개 파일을 생성/업로드합니다:
    1. **`app.py`**: 위의 파이썬 소스 코드를 그대로 붙여넣기 합니다.
    2. **`requirements.txt`**: 아래 패키지 목록을 파일로 추가합니다.
    ```text
    streamlit
    pandas
    ```

    ### 3단계: Streamlit Community Cloud 배포
    1. [share.streamlit.io](https://share.streamlit.io) 접속 및 깃허브 계정 연동
    2. **New app** 버튼 클릭
    3. 방금 만든 Repository, Branch(`main`), Main file path(`app.py`)를 지정하고 **Deploy!** 버튼 클릭
    4. 생성된 URL 주소를 학생들에게 공유하면 어디서나 접근 가능한 수업용 도구가 완성이 됩니다!
    """)
