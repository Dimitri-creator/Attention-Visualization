import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import GUI from 'lil-gui';
import './App.css';

// Visualization constants
const CUBE_SIZE = 0.8;
const CUBE_SPACING = 1.5;
const MATRIX_CELL_SIZE = 0.25;
const MATRIX_CELL_SPACING = 0.05;
const MATRIX_BASE_Y_OFFSET = CUBE_SIZE * 2.5;
const HEAD_VIS_SPACING_Y = MATRIX_CELL_SIZE * 18;
const HEAD_VIS_SPACING_X = MATRIX_CELL_SIZE * 12;
const SDPA_STAGE_SPACING_Z = MATRIX_CELL_SIZE * 10;
const CONCAT_FINAL_OFFSET_Y_FACTOR = 1.5;
const ADDNORM_VIS_OFFSET_Y_FACTOR = 1.0;
const FFN_VIS_OFFSET_Y_FACTOR = 1.5;
const HIGHLIGHT_COLOR = 0xffaa00;
const HIGHLIGHT_DURATION = 1500;
const SELECTED_HEAD_EMISSIVE_INTENSITY = 0.4;
const OTHER_HEAD_EMISSIVE_INTENSITY = 0.0;


type Matrix = number[][];
type StepName = "input" | "qkv_mha" | "sdpa_heads" | "concat_finalize" | "mha_addnorm" | "ffn" | "ffn_addnorm" | "none";


function App() {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2(-1000, -1000);
  let INTERSECTED: THREE.Object3D | null = null;
  const guiRef = useRef<GUI | null>(null); // Ref to store GUI instance for dynamic updates

  // --- Refs for various visualization groups ---
  const tokenEmbeddingVisGroupRef = useRef<THREE.Group>(new THREE.Group());
  const peVisGroupRef = useRef<THREE.Group>(new THREE.Group());
  const finalInputVisGroupRef = useRef<THREE.Group>(new THREE.Group());
  const wqMhaGroupRef = useRef<THREE.Group>(new THREE.Group());
  const wkMhaGroupRef = useRef<THREE.Group>(new THREE.Group());
  const wvMhaGroupRef = useRef<THREE.Group>(new THREE.Group());
  const qProjectedGroupRef = useRef<THREE.Group>(new THREE.Group());
  const kProjectedGroupRef = useRef<THREE.Group>(new THREE.Group());
  const vProjectedGroupRef = useRef<THREE.Group>(new THREE.Group());
  const qHeadsGroupRefs = useRef<THREE.Group[]>([]);
  const kHeadsGroupRefs = useRef<THREE.Group[]>([]);
  const vHeadsGroupRefs = useRef<THREE.Group[]>([]);
  const attentionScoresScaledHeadsGroupRefs = useRef<THREE.Group[]>([]);
  const attentionWeightsHeadsGroupRefs = useRef<THREE.Group[]>([]);
  const headOutputsGroupRefs = useRef<THREE.Group[]>([]);
  const concatenatedHeadsGroupRef = useRef<THREE.Group>(new THREE.Group());
  const woMhaGroupRef = useRef<THREE.Group>(new THREE.Group());
  const mhaFinalOutputGroupRef = useRef<THREE.Group>(new THREE.Group());
  const mhaResidualSumGroupRef = useRef<THREE.Group>(new THREE.Group());
  const mhaNormOutputGroupRef = useRef<THREE.Group>(new THREE.Group());
  const w1FfnGroupRef = useRef<THREE.Group>(new THREE.Group());
  const ffnIntermediateLinearGroupRef = useRef<THREE.Group>(new THREE.Group());
  const ffnReluOutputGroupRef = useRef<THREE.Group>(new THREE.Group());
  const w2FfnGroupRef = useRef<THREE.Group>(new THREE.Group());
  const ffnFinalOutputGroupRef = useRef<THREE.Group>(new THREE.Group());
  const ffnResidualSumGroupRef = useRef<THREE.Group>(new THREE.Group());
  const ffnNormOutputGroupRef = useRef<THREE.Group>(new THREE.Group());

  const allMatrixVisGroups = [ /* ... as before ... */ ];
  const dynamicMatrixGroupArrays = [ /* ... as before ... */ ];

  // --- State Variables ---
  const [lastCompletedStep, setLastCompletedStep] = useState<StepName>("none");
  const [selectedHeadForHighlight, setSelectedHeadForHighlight] = useState<number>(0); // New
  // ... (all other state variables as before)
  const [inputText, setInputText] = useState("hello,world");
  const [dModel, setDModel] = useState(8);
  const [numHeads, setNumHeads] = useState(2);
  const [dkPerHead, setDkPerHead] = useState(4);
  const [dff, setDff] = useState(32);
  const [tokens, setTokens] = useState<string[]>([]);
  const [finalInputMatrix, setFinalInputMatrix] = useState<Matrix>([]);
  const [WqMHA, setWqMHA] = useState<Matrix>([]); setWkMHA([]); setWvMHA([]);
  const [qProjected, setQProjected] = useState<Matrix>([]); setKProjected([]); setVProjected([]);
  const [qHeads, setQHeads] = useState<Matrix[]>([]); setKHeads([]); setVHeads([]);
  const [attentionScoresScaled_heads, setAttentionScoresScaled_heads] = useState<Matrix[]>([]);
  const [attentionWeights_heads, setAttentionWeights_heads] = useState<Matrix[]>([]);
  const [headOutputs, setHeadOutputs] = useState<Matrix[]>([]);
  const [concatenatedHeadOutputs, setConcatenatedHeadOutputs] = useState<Matrix>([]);
  const [Wo_mha, setWo_mha] = useState<Matrix>([]);
  const [mhaFinalOutput, setMhaFinalOutput] = useState<Matrix>([]);
  const [mhaResidualSum, setMhaResidualSum] = useState<Matrix>([]);
  const [gamma1, setGamma1] = useState<number[]>([]);
  const [beta1, setBeta1] = useState<number[]>([]);
  const [mhaNormOutput, setMhaNormOutput] = useState<Matrix>([]);
  const [W1_ffn, setW1_ffn] = useState<Matrix>([]); setB1_ffn([]); setW2_ffn([]); setB2_ffn([]);
  const [b1_ffn, setB1_ffn] = useState<Matrix>([]);
  const [W2_ffn, setW2_ffn] = useState<Matrix>([]);
  const [b2_ffn, setB2_ffn] = useState<Matrix>([]);
  const [ffnIntermediateLinear, setFfnIntermediateLinear] = useState<Matrix>([]);
  const [ffnReluOutput, setFfnReluOutput] = useState<Matrix>([]);
  const [ffnFinalOutput, setFfnFinalOutput] = useState<Matrix>([]);
  const [ffnResidualSum, setFfnResidualSum] = useState<Matrix>([]);
  const [gamma2, setGamma2] = useState<number[]>([]);
  const [beta2, setBeta2] = useState<number[]>([]);
  const [ffnNormOutput, setFfnNormOutput] = useState<Matrix>([]);

  // --- Matrix Utility Functions ---
  // ... (All matrix utilities as before)
  const generateRandomMatrix = (rows: number, cols: number): Matrix => Array.from({ length: rows }, () => Array.from({ length: cols }, () => parseFloat((Math.random() * 0.4 - 0.2).toFixed(4)) ));
  const matrixMultiply = (A: Matrix, B: Matrix): Matrix | null => { const rA = A.length; const cA = A[0]?.length||0; const rB = B.length; const cB = B[0]?.length||0; if(cA===0&&rA===0&&cB===0&&rB===0)return[]; if(cA===0&&rA===0)return Array(0).fill(Array(cB).fill(0)); if(cB===0&&rB===0)return Array(rA).fill(Array(0).fill(0)); if(cA!==rB){console.error(`Matrix mul dim mismatch: A_cols(${cA}) !== B_rows(${rB})`);return null;} const res:Matrix=[]; for(let i=0;i<rA;i++){res[i]=[];for(let j=0;j<cB;j++){let sum=0;for(let k=0;k<cA;k++){sum+=(A[i][k]||0)*(B[k][j]||0);}res[i][j]=sum;}} return res; };
  const transposeMatrix = (M: Matrix): Matrix => { const r=M.length; const c=M[0]?.length||0; const res:Matrix=[]; for(let j=0;j<c;j++){res[j]=[];for(let i=0;i<r;i++){res[j][i]=M[i][j];}} return res; };
  const scaleMatrix = (M: Matrix, s: number): Matrix => M.map(row => row.map(val => val * s));
  const elementWiseSumMatrices = (A: Matrix, B: Matrix): Matrix | null => { if(!A||!B||A.length!==B.length||(A[0]?.length||0)!==(B[0]?.length||0)){console.error("Element-wise sum dim mismatch or invalid matrices.",A,B);return null;} return A.map((r,i)=>r.map((v,j)=>v+(B[i][j]||0))); };
  const applySoftmax = (M: Matrix): Matrix => M.map(r=>{if(r.length===0)return[];const max=Math.max(...r);const exps=r.map(v=>Math.exp(v-max));const sum=exps.reduce((s,v)=>s+v,0);if(sum===0)return r.map(()=>1/r.length);return exps.map(v=>v/sum);});
  const addBias = (matrix: Matrix, biasVector: number[]): Matrix => { if(!matrix||matrix.length===0||!biasVector||biasVector.length===0)return matrix; if(matrix[0].length!==biasVector.length){console.error("Bias vector length must match matrix column count.");return matrix;} return matrix.map(r=>r.map((v,j)=>v+biasVector[j]));};
  const applyRelu = (matrix: Matrix): Matrix => matrix.map(row => row.map(val => Math.max(0, val)));
  const applyLayerNorm = (matrix: Matrix, gamma: number[], beta: number[], epsilon: number = 1e-5): Matrix => { if(!matrix||matrix.length===0)return[]; return matrix.map(r=>{if(r.length===0)return[];const mean=r.reduce((a,v)=>a+v,0)/r.length;const vari=r.reduce((a,v)=>a+Math.pow(v-mean,2),0)/r.length;const invStd=1/Math.sqrt(vari+epsilon);return r.map((v,j)=>(gamma[j]*(v-mean)*invStd)+beta[j]);});};

  // --- Visualization Functions ---
  // ... (clearGroup, visualizeSimplifiedVectors, visualizeMatrixAsGrid, highlightGroupTemporarily as before)
  const clearGroup = (group: THREE.Group) => { while(group.children.length>0){const c=group.children[0];group.remove(c);if(c instanceof THREE.Mesh){c.geometry.dispose();if(Array.isArray(c.material)){c.material.forEach(m=>m.dispose());}else{c.material.dispose();}}}}};
  const visualizeSimplifiedVectors = (vectors: Matrix, color: THREE.ColorRepresentation, yBaseOffset: number = 0, groupToUpdate: THREE.Group): void => { clearGroup(groupToUpdate); vectors.forEach((vec, index) => { const firstComponent = vec[0] || 0; const height = Math.max(0.1, Math.abs(firstComponent) * CUBE_SIZE + 0.1); const geometry = new THREE.BoxGeometry(CUBE_SIZE, height, CUBE_SIZE); const material = new THREE.MeshStandardMaterial({ color }); const cube = new THREE.Mesh(geometry, material); cube.position.set(index * CUBE_SPACING - (vectors.length * CUBE_SPACING / 2) + CUBE_SPACING / 2, yBaseOffset + height / 2, 0); groupToUpdate.add(cube); }); };
  const visualizeMatrixAsGridReal = ( matrix: Matrix, group: THREE.Group, positionOffset: THREE.Vector3, baseColor: THREE.ColorRepresentation = 0xcccccc, normalize: boolean = true, fixedHeight: number | null = null, matrixName?: string) => { clearGroup(group); if (!matrix || matrix.length === 0 || matrix[0].length === 0) return; const rows = matrix.length; const cols = matrix[0].length; let minVal = Infinity, maxVal = -Infinity; if (normalize && fixedHeight === null) { matrix.forEach(row => row.forEach(val => { if (val < minVal) minVal = val; if (val > maxVal) maxVal = val; })); if (minVal === maxVal) maxVal = minVal + 1; } for (let i = 0; i < rows; i++) { for (let j = 0; j < cols; j++) { const value = matrix[i][j]; let heightScale = value; if (fixedHeight !== null) { heightScale = value; } else if (normalize) { heightScale = (value - minVal) / (maxVal - minVal); } const cellHeight = fixedHeight !== null ? Math.max(0.01, heightScale * fixedHeight) : Math.max(0.05, heightScale * MATRIX_CELL_SIZE * 2); const geometry = new THREE.BoxGeometry(MATRIX_CELL_SIZE, cellHeight, MATRIX_CELL_SIZE); const material = new THREE.MeshStandardMaterial({ color: baseColor }); const cube = new THREE.Mesh(geometry, material); cube.name = `${matrixName}_cell_${i}_${j}`; if(matrixName) cube.userData = { matrixName, row: i, col: j, value: parseFloat(value.toFixed(4)) }; cube.position.set( j * (MATRIX_CELL_SIZE + MATRIX_CELL_SPACING) - (cols * (MATRIX_CELL_SIZE + MATRIX_CELL_SPACING) / 2) + MATRIX_CELL_SPACING/2, cellHeight / 2,  i * (MATRIX_CELL_SIZE + MATRIX_CELL_SPACING) - (rows * (MATRIX_CELL_SIZE + MATRIX_CELL_SPACING) / 2) + MATRIX_CELL_SPACING/2); group.add(cube); } } group.position.copy(positionOffset);}; visualizeMatrixAsGrid = visualizeMatrixAsGridReal;
  const highlightGroupTemporarily = (groupRef: React.RefObject<THREE.Group>, highlightColor: THREE.ColorRepresentation = HIGHLIGHT_COLOR, duration: number = HIGHLIGHT_DURATION) => { if (!groupRef.current) return; const originalMaterials = new Map<THREE.Mesh, THREE.Material | THREE.Material[]>(); groupRef.current.traverse((child) => { if (child instanceof THREE.Mesh && child.material) { originalMaterials.set(child, child.material); if (Array.isArray(child.material)) { /* TODO */ } else { const highlightMaterial = (child.material as THREE.MeshStandardMaterial).clone(); (highlightMaterial as THREE.MeshStandardMaterial).emissive = new THREE.Color(highlightColor); (highlightMaterial as THREE.MeshStandardMaterial).emissiveIntensity = 0.6; child.material = highlightMaterial; } } }); setTimeout(() => { groupRef.current?.traverse((child) => { if (child instanceof THREE.Mesh && originalMaterials.has(child)) { const originalMaterial = originalMaterials.get(child); if (originalMaterial) child.material = originalMaterial; } }); }, duration); };

  // --- Core Logic ---
  // ... (All handle... functions from previous step, with highlightGroupTemporarily calls)
  const generateFullTokenEmbeddings = (s: string[],cD: number): Matrix => Array.from({length:s.length},()=>Array.from({length:cD},()=>parseFloat((Math.random()*0.5-0.25).toFixed(4))));
  const generateFullPositionalEncodings = (sL: number,cD: number): Matrix => {const pe:Matrix=[];for(let p=0;p<sL;p++){const comp:number[]=[];for(let i=0;i<cD;i++){comp.push(parseFloat(((i%2===0)?Math.sin(p/Math.pow(10000,(2*i)/cD)):Math.cos(p/Math.pow(10000,(2*(i-1))/cD))).toFixed(4)));}pe.push(comp);}return pe;};
  const updateInfoPanel = (message: string, status: "info" | "error" = "info") => { const ipEl=document.getElementById('info-panel'); if(ipEl){const base=`Tokens: ${tokens.join(', ')||'N/A'} | d_m: ${dModel} | H: ${numHeads} | d_k: ${dkPerHead} | d_ff: ${dff}`; ipEl.innerHTML=`<h4>Info</h4><p>${base}</p><p style="color:${status==="error"?"#ffaaaa":"#aaffaa"};">${message}</p><p style="font-size:0.8em;color:#ccc;">Last completed: ${lastCompletedStep}</p>`;}};
  const splitHeads = (pM: Matrix,cN: number,cDk: number): Matrix[] => { if(!pM||pM.length===0)return[];const sL=pM.length;const h:Matrix[]=Array.from({length:cN},()=>Array.from({length:sL},()=>Array(cDk).fill(0)));for(let i=0;i<cN;i++){for(let j=0;j<sL;j++){for(let k=0;k<cDk;k++){h[i][j][k]=pM[j][i*cDk+k];}}}return h;};
  const concatenateHeads = (hOM: Matrix[],cN: number,cDk: number): Matrix => { if(!hOM||hOM.length===0||hOM.length!==cN)return[];const sL=hOM[0].length;if(sL===0)return[];const cDM=cN*cDk;const res:Matrix=Array.from({length:sL},()=>Array(cDM).fill(0));for(let i=0;i<sL;i++){let cC=0;for(let j=0;j<cN;j++){for(let l=0;l<cDk;l++){res[i][cC++]=hOM[j][i][l];}}}return res;};
  const manageHeadGroupsArray = (hGAR: React.MutableRefObject<THREE.Group[]>,cnt: number,scn: THREE.Scene) => {while(hGAR.current.length>cnt){const g=hGAR.current.pop();if(g){clearGroup(g);scn.remove(g);}}while(hGAR.current.length<cnt){const nG=new THREE.Group();hGAR.current.push(nG);scn.add(nG);}};
  const stepOrder: StepName[] = ["input", "qkv_mha", "sdpa_heads", "concat_finalize", "mha_addnorm", "ffn", "ffn_addnorm"];
  const isStepEnabled = (stepName: StepName): boolean => { if (stepName === "input") return true; const lastIdx = stepOrder.indexOf(lastCompletedStep); const currentIdx = stepOrder.indexOf(stepName); return lastIdx >= currentIdx -1; };
  const clearSubsequentSteps = (currentStep: StepName) => { /* ... */ };
  const updateDkPerHeadInput = (cD: number,cNH: number,cDff?: number) => { /* ... */ };
  const handleGenerateInputEmbeddings = () => { /* ... */ setLastCompletedStep("input"); updateInfoPanel("Generated input.", "info"); highlightGroupTemporarily(finalInputVisGroupRef);};
  const handleCalculateQKV_MHA = () => { /* ... */ setLastCompletedStep("qkv_mha"); updateInfoPanel("MHA QKV calculated.", "info"); /* highlight relevant groups */};
  const handleRunAllHeadsSDPA = () => { /* ... */ setLastCompletedStep("sdpa_heads"); updateInfoPanel("SDPA for all heads calculated.", "info"); /* highlight relevant groups */};
  const handleConcatenateAndFinalizeMHA = () => { /* ... */ setLastCompletedStep("concat_finalize"); updateInfoPanel("Heads concatenated & projected.", "info"); /* highlight relevant groups */};
  const handleAddNormMHA = () => { /* ... */ setLastCompletedStep("mha_addnorm"); updateInfoPanel("Add & Norm after MHA.", "info"); /* highlight relevant groups */};
  const handleRunFFN = () => { /* ... */ setLastCompletedStep("ffn"); updateInfoPanel("FFN executed.", "info"); /* highlight relevant groups */};
  const handleAddNormFFN = () => { /* ... */ setLastCompletedStep("ffn_addnorm"); updateInfoPanel("Encoder Layer Complete!", "info"); /* highlight relevant groups */};
  const clearSubsequentStepsReal = (currentStep: StepName) => { const currentStepIdx=stepOrder.indexOf(currentStep); const stepsToClear=stepOrder.slice(currentStepIdx+1); const clearActions:Record<StepName,()=>void>={"none":()=>{},"input":()=>{},"qkv_mha":()=>{setAttentionScoresScaled_heads([]);setAttentionWeights_heads([]);setHeadOutputs([]);attentionScoresScaledHeadsGroupRefs.current.forEach(g=>clearGroup(g));attentionWeightsHeadsGroupRefs.current.forEach(g=>clearGroup(g));headOutputsGroupRefs.current.forEach(g=>clearGroup(g));},"sdpa_heads":()=>{setConcatenatedHeadOutputs([]);setWo_mha([]);setMhaFinalOutput([]);clearGroup(concatenatedHeadsGroupRef.current);clearGroup(woMhaGroupRef.current);clearGroup(mhaFinalOutputGroupRef.current);},"concat_finalize":()=>{setMhaResidualSum([]);setMhaNormOutput([]);clearGroup(mhaResidualSumGroupRef.current);clearGroup(mhaNormOutputGroupRef.current);},"mha_addnorm":()=>{setW1_ffn([]);setB1_ffn([]);setW2_ffn([]);setB2_ffn([]);setFfnIntermediateLinear([]);setFfnReluOutput([]);setFfnFinalOutput([]);clearGroup(w1FfnGroupRef.current);clearGroup(ffnIntermediateLinearGroupRef.current);clearGroup(ffnReluOutputGroupRef.current);clearGroup(w2FfnGroupRef.current);clearGroup(ffnFinalOutputGroupRef.current);},"ffn":()=>{setFfnResidualSum([]);setFfnNormOutput([]);clearGroup(ffnResidualSumGroupRef.current);clearGroup(ffnNormOutputGroupRef.current);},"ffn_addnorm":()=>{}}; if(currentStep==="input"){setWqMHA([]);setWkMHA([]);setWvMHA([]);setQProjected([]);setKProjected([]);setVProjected([]);setQHeads([]);setKHeads([]);setVHeads([]);clearGroup(wqMhaGroupRef.current);clearGroup(wkMhaGroupRef.current);clearGroup(wvMhaGroupRef.current);clearGroup(qProjectedGroupRef.current);clearGroup(kProjectedGroupRef.current);clearGroup(vProjectedGroupRef.current);qHeadsGroupRefs.current.forEach(g=>clearGroup(g));kHeadsGroupRefs.current.forEach(g=>clearGroup(g));vHeadsGroupRefs.current.forEach(g=>clearGroup(g));stepsToClear.forEach(step=>clearActions[step]?.());}else{stepsToClear.forEach(step=>clearActions[step]?.());}}; clearSubsequentSteps=clearSubsequentStepsReal;
  const updateDkPerHeadInputReal = (cD:number,cNH:number,cDff?:number)=>{const dkIE=document.getElementById('dk-input')as HTMLInputElement;if(dkIE){if(cD>0&&cNH>0&&cD%cNH===0){const v=cD/cNH;dkIE.value=v.toString();setDkPerHead(v);dkIE.style.backgroundColor="#555";}else{dkIE.value="N/A";dkIE.style.backgroundColor="#855";}}const dffIE=document.getElementById('dff-input')as HTMLInputElement;if(dffIE){const nDff=cD*4;if(dffIE.value!==String(nDff)&&(cDff===undefined||cDff===0||dffIE.value===String((cD/2)*4))){dffIE.value=String(nDff);setDff(nDff);}else if(cDff!==undefined){setDff(cDff);}}}; updateDkPerHeadInput=updateDkPerHeadInputReal;
  const handleGenerateInputEmbeddingsReal = () => { const tIE=document.getElementById('token-input')as HTMLInputElement;const dMIE=document.getElementById('dmodel-input')as HTMLInputElement;const nHIE=document.getElementById('num-heads-input')as HTMLInputElement;const dffIE=document.getElementById('dff-input')as HTMLInputElement;const cIT=tIE.value;const cDM=parseInt(dMIE.value);const cNH=parseInt(nHIE.value);const cDff=parseInt(dffIE.value);setInputText(cIT);setDModel(cDM);setNumHeads(cNH);setDff(cDff);updateDkPerHeadInput(cDM,cNH,cDff);const cT=cIT.split(',').map(t=>t.trim()).filter(t=>t);setTokens(cT);const nE=generateFullTokenEmbeddings(cT,cDM);const nPE=generateFullPositionalEncodings(cT.length,cDM);const nFI=elementWiseSumMatrices(nE,nPE);if(nFI)setFinalInputMatrix(nFI);visualizeSimplifiedVectors(nE,0x00ff00,0,tokenEmbeddingVisGroupRef.current);visualizeSimplifiedVectors(nPE,0xff0000,CUBE_SIZE*1.5,peVisGroupRef.current);if(nFI)visualizeSimplifiedVectors(nFI,0x0000ff,CUBE_SIZE*-1.5,finalInputVisGroupRef.current);clearSubsequentSteps("input");setLastCompletedStep("input");updateInfoPanel(`Generated input for ${cT.length} tokens.`); highlightGroupTemporarily(finalInputVisGroupRef);}; handleGenerateInputEmbeddings=handleGenerateInputEmbeddingsReal;
  const handleCalculateQKV_MHAReal = () => { const dMV=dModel;const nHV=numHeads;if(dMV%nHV!==0){updateInfoPanel(`d_model (${dMV}) must be divisible by Num Heads (${nHV}).`,"error");return;}const cDkPHV=dMV/nHV;setDkPerHead(cDkPHV);if(finalInputMatrix.length===0){updateInfoPanel("Generate input first.","error");return;}clearSubsequentSteps("qkv_mha");const nWq=generateRandomMatrix(dMV,dMV);setWqMHA(nWq);const nWk=generateRandomMatrix(dMV,dMV);setWkMHA(nWk);const nWv=generateRandomMatrix(dMV,dMV);setWvMHA(nWv);const mhaWOX=(dMV*MATRIX_CELL_SIZE*1.2+1);visualizeMatrixAsGrid(nWq,wqMhaGroupRef.current,new THREE.Vector3(-mhaWOX,MATRIX_BASE_Y_OFFSET,0),0xff8844,true,null,"Wq_MHA");visualizeMatrixAsGrid(nWk,wkMhaGroupRef.current,new THREE.Vector3(0,MATRIX_BASE_Y_OFFSET,0),0x44ff88,true,null,"Wk_MHA");visualizeMatrixAsGrid(nWv,wvMhaGroupRef.current,new THREE.Vector3(mhaWOX,MATRIX_BASE_Y_OFFSET,0),0x8844ff,true,null,"Wv_MHA");const qp=matrixMultiply(finalInputMatrix,nWq);setQProjected(qp||[]);const kp=matrixMultiply(finalInputMatrix,nWk);setKProjected(kp||[]);const vp=matrixMultiply(finalInputMatrix,nWv);setVProjected(vp||[]);const pVOY=MATRIX_BASE_Y_OFFSET+(dMV*MATRIX_CELL_SIZE*1.2+0.5);if(qp)visualizeMatrixAsGrid(qp,qProjectedGroupRef.current,new THREE.Vector3(-mhaWOX,pVOY,0),0xffccaa,true,null,"Q_Projected");if(kp)visualizeMatrixAsGrid(kp,kProjectedGroupRef.current,new THREE.Vector3(0,pVOY,0),0xaaccff,true,null,"K_Projected");if(vp)visualizeMatrixAsGrid(vp,vProjectedGroupRef.current,new THREE.Vector3(mhaWOX,pVOY,0),0xccaaff,true,null,"V_Projected");if(qp&&kp&&vp){const qh=splitHeads(qp,nHV,cDkPHV);setQHeads(qh);const kh=splitHeads(kp,nHV,cDkPHV);setKHeads(kh);const vh=splitHeads(vp,nHV,cDkPHV);setVHeads(vh);if(sceneRef.current){manageHeadGroupsArray(qHeadsGroupRefs,nHV,sceneRef.current!);manageHeadGroupsArray(kHeadsGroupRefs,nHV,sceneRef.current!);manageHeadGroupsArray(vHeadsGroupRefs,nHV,sceneRef.current!);}const hVBOY=pVOY+(dMV*MATRIX_CELL_SIZE*1.2+1);for(let h=0;h<nHV;h++){const hOY=hVBOY+h*HEAD_VIS_SPACING_Y;if(qHeadsGroupRefs.current[h]&&qh[h])visualizeMatrixAsGrid(qh[h],qHeadsGroupRefs.current[h],new THREE.Vector3(-HEAD_VIS_SPACING_X,hOY,0),0xffaaaa,true,null,`Q_Head_${h}`);if(kHeadsGroupRefs.current[h]&&kh[h])visualizeMatrixAsGrid(kh[h],kHeadsGroupRefs.current[h],new THREE.Vector3(0,hOY,0),0xaaffaa,true,null,`K_Head_${h}`);if(vHeadsGroupRefs.current[h]&&vh[h])visualizeMatrixAsGrid(vh[h],vHeadsGroupRefs.current[h],new THREE.Vector3(HEAD_VIS_SPACING_X,hOY,0),0xaaaaff,true,null,`V_Head_${h}`);}setLastCompletedStep("qkv_mha");updateInfoPanel(`MHA Q,K,V (H:${nHV},dk/h:${cDkPHV}).`); highlightGroupTemporarily(qProjectedGroupRef); highlightGroupTemporarily(kProjectedGroupRef); highlightGroupTemporarily(vProjectedGroupRef);}else{updateInfoPanel("Error projecting Q,K,V.","error");}}; handleCalculateQKV_MHA=handleCalculateQKV_MHAReal;
  const handleRunAllHeadsSDPAReal = () => { if(qHeads.length===0||kHeads.length===0||vHeads.length===0||qHeads.length!==numHeads){updateInfoPanel("MHA Q,K,V not calculated.","error");return;}clearSubsequentSteps("sdpa_heads");const cDk=dkPerHead;const nSS_h:Matrix[]=[];const nW_h:Matrix[]=[];const nO_h:Matrix[]=[];if(sceneRef.current){manageHeadGroupsArray(attentionScoresScaledHeadsGroupRefs,numHeads,sceneRef.current!);manageHeadGroupsArray(attentionWeightsHeadsGroupRefs,numHeads,sceneRef.current!);manageHeadGroupsArray(headOutputsGroupRefs,numHeads,sceneRef.current!);}const qkvPVOffsetY=MATRIX_BASE_Y_OFFSET+(dModel*MATRIX_CELL_SIZE*1.2+0.5);const hQKVBaseOffsetY=qkvPVOffsetY+(dModel*MATRIX_CELL_SIZE*1.2+1);for(let h=0;h<numHeads;h++){const Q_h=qHeads[h];const K_h=kHeads[h];const V_h=vHeads[h];if(!Q_h||!K_h||!V_h){console.error(`Data for head ${h} missing.`);continue;}const KT_h=transposeMatrix(K_h);const RS_h=matrixMultiply(Q_h,KT_h);if(!RS_h){console.error(`Error RawScores head ${h}`);continue;}const SS_h=scaleMatrix(RS_h,1/Math.sqrt(cDk));nSS_h[h]=SS_h;const hSY_QKV=hQKVBaseOffsetY+h*HEAD_VIS_SPACING_Y;if(attentionScoresScaledHeadsGroupRefs.current[h]){visualizeMatrixAsGrid(SS_h,attentionScoresScaledHeadsGroupRefs.current[h],new THREE.Vector3(-HEAD_VIS_SPACING_X,hSY_QKV,SDPA_STAGE_SPACING_Z),0xffdddd,true,null,`Scores_H${h}`); highlightGroupTemporarily(attentionScoresScaledHeadsGroupRefs.current[h]);}const W_h=applySoftmax(SS_h);nW_h[h]=W_h;if(attentionWeightsHeadsGroupRefs.current[h]){visualizeMatrixAsGrid(W_h,attentionWeightsHeadsGroupRefs.current[h],new THREE.Vector3(0,hSY_QKV,SDPA_STAGE_SPACING_Z),0xddffdd,false,MATRIX_CELL_SIZE*1.5,`Weights_H${h}`); highlightGroupTemporarily(attentionWeightsHeadsGroupRefs.current[h]);}const O_h=matrixMultiply(W_h,V_h);if(!O_h){console.error(`Error Output_h head ${h}`);continue;}nO_h[h]=O_h;if(headOutputsGroupRefs.current[h]){visualizeMatrixAsGrid(O_h,headOutputsGroupRefs.current[h],new THREE.Vector3(HEAD_VIS_SPACING_X,hSY_QKV,SDPA_STAGE_SPACING_Z),0xddddff,true,null,`Output_H${h}`); highlightGroupTemporarily(headOutputsGroupRefs.current[h]);}}setAttentionScoresScaled_heads(nSS_h);setAttentionWeights_heads(nW_h);setHeadOutputs(nO_h);setLastCompletedStep("sdpa_heads");updateInfoPanel(`SDPA for ${numHeads} heads.`);}; handleRunAllHeadsSDPA=handleRunAllHeadsSDPAReal;
  const handleConcatenateAndFinalizeMHAReal = () => { if(headOutputs.length!==numHeads||headOutputs.some(h=>!h||h.length===0)){updateInfoPanel("Per-head SDPA outputs not available.","error");return;}clearSubsequentSteps("concat_finalize");const concat=concatenateHeads(headOutputs,numHeads,dkPerHead);setConcatenatedHeadOutputs(concat);const lHSdpaY=MATRIX_BASE_Y_OFFSET+(dModel*MATRIX_CELL_SIZE*1.2+0.5)*2+1+(numHeads-1)*HEAD_VIS_SPACING_Y; const concatVisY=lHSdpaY+HEAD_VIS_SPACING_Y*CONCAT_FINAL_OFFSET_Y_FACTOR+SDPA_STAGE_SPACING_Z;if(concat.length>0){visualizeMatrixAsGrid(concat,concatenatedHeadsGroupRef.current,new THREE.Vector3(0,concatVisY,0),0xffffaa,"Concat_Heads");highlightGroupTemporarily(concatenatedHeadsGroupRef);}const cDMV=numHeads*dkPerHead;const nWo=generateRandomMatrix(cDMV,cDMV);setWo_mha(nWo);visualizeMatrixAsGrid(nWo,woMhaGroupRef.current,new THREE.Vector3((cDMV*MATRIX_CELL_SIZE*0.7+1),concatVisY,0),0x00ffff,"Wo_MHA");const finOut=matrixMultiply(concat,nWo);if(finOut){setMhaFinalOutput(finOut);visualizeMatrixAsGrid(finOut,mhaFinalOutputGroupRef.current,new THREE.Vector3(0,concatVisY+HEAD_VIS_SPACING_Y*0.7,0),0xffaaff,"MHA_Output");highlightGroupTemporarily(mhaFinalOutputGroupRef);setLastCompletedStep("concat_finalize");updateInfoPanel("Heads concatenated & projected.");}else{updateInfoPanel("Error in final MHA projection.","error");}}; handleConcatenateAndFinalizeMHA=handleConcatenateAndFinalizeMHAReal;
  const handleAddNormMHAReal = () => { if(!mhaFinalOutput||mhaFinalOutput.length===0||!finalInputMatrix||finalInputMatrix.length===0){updateInfoPanel("MHA output or original input missing.","error");return;}if(mhaFinalOutput.length!==finalInputMatrix.length||mhaFinalOutput[0].length!==finalInputMatrix[0].length){updateInfoPanel("Dimension mismatch MHA AddNorm.","error");return;}clearSubsequentSteps("mha_addnorm");const resSum=elementWiseSumMatrices(finalInputMatrix,mhaFinalOutput);if(!resSum){updateInfoPanel("Error MHA residual sum.","error");return;}setMhaResidualSum(resSum);const cDMV=dModel;const nG1=Array(cDMV).fill(1);setGamma1(nG1);const nB1=Array(cDMV).fill(0);setBeta1(nB1);const normOut=applyLayerNorm(resSum,nG1,nB1);setMhaNormOutput(normOut);const mhaWoOutY=mhaFinalOutputGroupRef.current.position.y;const resVY=mhaWoOutY+HEAD_VIS_SPACING_Y*ADDNORM_VIS_OFFSET_Y_FACTOR;visualizeMatrixAsGrid(resSum,mhaResidualSumGroupRef.current,new THREE.Vector3(-HEAD_VIS_SPACING_X*0.5,resVY,0),0xffd700,"MHA_ResSum");highlightGroupTemporarily(mhaResidualSumGroupRef);visualizeMatrixAsGrid(normOut,mhaNormOutputGroupRef.current,new THREE.Vector3(HEAD_VIS_SPACING_X*0.5,resVY,0),0x7cfc00,"MHA_NormOut");highlightGroupTemporarily(mhaNormOutputGroupRef);setLastCompletedStep("mha_addnorm");updateInfoPanel("Add & Norm after MHA.");}; handleAddNormMHA=handleAddNormMHAReal;
  const handleRunFFNReal = () => { const ffnIn=(mhaNormOutput&&mhaNormOutput.length>0)?mhaNormOutput:mhaFinalOutput;if(!ffnIn||ffnIn.length===0){updateInfoPanel("Input for FFN not available.","error");return;}clearSubsequentSteps("ffn");const dffVal=parseInt((document.getElementById('dff-input')as HTMLInputElement).value)||(dModel*4);setDff(dffVal);const cDMV=ffnIn[0].length;const nW1=generateRandomMatrix(cDMV,dffVal);setW1_ffn(nW1);const nB1=generateRandomMatrix(1,dffVal);setB1_ffn(nB1);const nW2=generateRandomMatrix(dffVal,cDMV);setW2_ffn(nW2);const nB2=generateRandomMatrix(1,cDMV);setB2_ffn(nB2);const pBY=mhaNormOutputGroupRef.current.children.length>0?mhaNormOutputGroupRef.current.position.y:mhaFinalOutputGroupRef.current.position.y;const ffNBY=pBY+HEAD_VIS_SPACING_Y*FFN_VIS_OFFSET_Y_FACTOR;visualizeMatrixAsGrid(nW1,w1FfnGroupRef.current,new THREE.Vector3(-HEAD_VIS_SPACING_X*0.7,ffNBY,0),0xffcccc,"W1_FFN");const l1nb=matrixMultiply(ffnIn,nW1);if(!l1nb){updateInfoPanel("Error FFN L1 multiply.","error");return;}const intLin=addBias(l1nb,nB1[0]);setFfnIntermediateLinear(intLin);visualizeMatrixAsGrid(intLin,ffnIntermediateLinearGroupRef.current,new THREE.Vector3(0,ffNBY,0),0xccffcc,"FFN_L1_Bias");highlightGroupTemporarily(ffnIntermediateLinearGroupRef);const rOut=applyRelu(intLin);setFfnReluOutput(rOut);visualizeMatrixAsGrid(rOut,ffnReluOutputGroupRef.current,new THREE.Vector3(0,ffNBY+HEAD_VIS_SPACING_Y*0.6,0),0xccccff,"FFN_ReLU");highlightGroupTemporarily(ffnReluOutputGroupRef);visualizeMatrixAsGrid(nW2,w2FfnGroupRef.current,new THREE.Vector3(HEAD_VIS_SPACING_X*0.7,ffNBY+HEAD_VIS_SPACING_Y*0.6,0),0xccffff,"W2_FFN");const l2nb=matrixMultiply(rOut,nW2);if(!l2nb){updateInfoPanel("Error FFN L2 multiply.","error");return;}const finFFNOut=addBias(l2nb,nB2[0]);setFfnFinalOutput(finFFNOut);visualizeMatrixAsGrid(finFFNOut,ffnFinalOutputGroupRef.current,new THREE.Vector3(0,ffNBY+HEAD_VIS_SPACING_Y*1.2,0),0xffccff,"FFN_Output");highlightGroupTemporarily(ffnFinalOutputGroupRef);setLastCompletedStep("ffn");updateInfoPanel(`FFN executed (d_ff:${dffVal}).`);}; handleRunFFN=handleRunFFNReal;
  const handleAddNormFFNReal = () => { let xFFNIn:Matrix|null=null;if(mhaNormOutput&&mhaNormOutput.length>0){xFFNIn=mhaNormOutput;}else if(mhaFinalOutput&&mhaFinalOutput.length>0){xFFNIn=mhaFinalOutput;}if(!ffnFinalOutput||ffnFinalOutput.length===0||!xFFNIn||xFFNIn.length===0){updateInfoPanel("FFN output or its input missing.","error");return;}if(ffnFinalOutput.length!==xFFNIn.length||ffnFinalOutput[0].length!==xFFNIn[0].length){updateInfoPanel("Dimension mismatch FFN AddNorm.","error");return;}clearSubsequentSteps("ffn_addnorm");const resSum=elementWiseSumMatrices(xFFNIn,ffnFinalOutput);if(!resSum){updateInfoPanel("Error FFN residual sum.","error");return;}setFfnResidualSum(resSum);const cDMV=dModel;const nG2=Array(cDMV).fill(1);setGamma2(nG2);const nB2=Array(cDMV).fill(0);setBeta2(nB2);const normOut=applyLayerNorm(resSum,nG2,nB2);setFfnNormOutput(normOut);const ffnOutY=ffnFinalOutputGroupRef.current.position.y;const resVY=ffnOutY+HEAD_VIS_SPACING_Y*ADDNORM_VIS_OFFSET_Y_FACTOR;visualizeMatrixAsGrid(resSum,ffnResidualSumGroupRef.current,new THREE.Vector3(-HEAD_VIS_SPACING_X*0.5,resVY,0),0xffd700,"FFN_ResSum");highlightGroupTemporarily(ffnResidualSumGroupRef);visualizeMatrixAsGrid(normOut,ffnNormOutputGroupRef.current,new THREE.Vector3(HEAD_VIS_SPACING_X*0.5,resVY,0),0x7cfc00,"Encoder_Output");highlightGroupTemporarily(ffnNormOutputGroupRef);setLastCompletedStep("ffn_addnorm");updateInfoPanel("Encoder Layer Complete!");}; handleAddNormFFN=handleAddNormFFNReal;

  // --- useEffect Hooks ---
  useEffect(() => {
    if (!mountRef.current || !sceneRef.current) return;
    tooltipRef.current = document.getElementById('tooltip') as HTMLDivElement;
    const guiEl = document.getElementById('gui-container');

    if (guiRef.current) { // Destroy old GUI if it exists to avoid duplication
        guiRef.current.destroy();
    }
    guiRef.current = new GUI({ container: guiEl!, title: "Transformer Encoder Layer Viz" });
    const gui = guiRef.current;

    const inputFolder = gui.addFolder('Input Params');
    const dModelCtrl = inputFolder.add({val: dModel}, 'val', 4, 512, 4).name('d_model').onChange(val => { (document.getElementById('dmodel-input') as HTMLInputElement).value = String(val); setDModel(val); updateDkPerHeadInput(val, numHeads, dff); });
    const numHeadsCtrl = inputFolder.add({val: numHeads}, 'val', 1, 8, 1).name('Num Heads').onChange(val => { (document.getElementById('num-heads-input') as HTMLInputElement).value = String(val); setNumHeads(val); updateDkPerHeadInput(dModel, val, dff); });
    inputFolder.add({ get val() { return (document.getElementById('dk-input') as HTMLInputElement)?.value; } }, 'val').name('d_k/head').listen();
    const dffCtrl = inputFolder.add({val: dff}, 'val', 4, 4096, 4).name('d_ff (FFN)').onChange(val => { (document.getElementById('dff-input') as HTMLInputElement).value = String(val); setDff(val); });

    const headHighlightOptions = Array.from({length: numHeads}, (_, i) => i);
    inputFolder.add({selectedHead: selectedHeadForHighlight}, 'selectedHead', headHighlightOptions).name('Highlight Head').onChange(setSelectedHeadForHighlight);


    const actionsFolder = gui.addFolder('Actions');
    const actionButtonIds: StepName[] = ["input", "qkv_mha", "sdpa_heads", "concat_finalize", "mha_addnorm", "ffn", "ffn_addnorm"];
    const actionHandlers = [handleGenerateInputEmbeddings, handleCalculateQKV_MHA, handleRunAllHeadsSDPA, handleConcatenateAndFinalizeMHA, handleAddNormMHA, handleRunFFN, handleAddNormFFN];
    const actionNames = ["1. Gen Input", "2. MHA QKV", "3. All SDPA", "4. MHA Concat", "5. AddNorm MHA", "6. FFN", "7. AddNorm FFN"];
    // actionsFolder.controllers.slice().forEach(ctrl => ctrl.destroy()); // Not needed due to full GUI destroy
    actionButtonIds.forEach((stepName, index) => { actionsFolder.add({ fn: actionHandlers[index] }, 'fn').name(actionNames[index]); });

    (document.getElementById('dmodel-input') as HTMLInputElement).onchange = (e) => dModelCtrl.setValue(parseInt((e.target as HTMLInputElement).value));
    (document.getElementById('num-heads-input') as HTMLInputElement).onchange = (e) => numHeadsCtrl.setValue(parseInt((e.target as HTMLInputElement).value));
    (document.getElementById('dff-input') as HTMLInputElement).onchange = (e) => dffCtrl.setValue(parseInt((e.target as HTMLInputElement).value));

    const htmlButtonIds: Record<StepName, string> = { "input": "generate-button", "qkv_mha": "calculate-qkv-button", "sdpa_heads": "run-all-heads-sdpa-button", "concat_finalize": "concatenate-finalize-mha-button", "mha_addnorm": "add-norm-mha-button", "ffn": "run-ffn-button", "ffn_addnorm": "add-norm-ffn-button", "none":"" };
    stepOrder.forEach((stepName) => { const buttonId = htmlButtonIds[stepName]; if (!buttonId) return; const button = document.getElementById(buttonId) as HTMLButtonElement; if (button) { button.disabled = !isStepEnabled(stepName); const handlerIndex = stepOrder.indexOf(stepName); button.removeEventListener('click', htmlActionHandlers[handlerIndex]); button.addEventListener('click', htmlActionHandlers[handlerIndex]); } });
    (document.getElementById('calculate-attention-button')as HTMLElement).style.display='none';(document.getElementById('calculate-output-button')as HTMLElement).style.display='none';
    if(lastCompletedStep === "none") { updateDkPerHeadInput(dModel,numHeads,dff); handleGenerateInputEmbeddings(); }
    return () => { guiRef.current?.destroy(); stepOrder.forEach((stepName) => { const buttonId = htmlButtonIds[stepName]; if(!buttonId) return; const b=document.getElementById(buttonId); const handlerIndex = stepOrder.indexOf(stepName); b?.removeEventListener('click',htmlActionHandlers[handlerIndex]); });};
  }, [dModel, numHeads, dff, lastCompletedStep, selectedHeadForHighlight]); // selectedHeadForHighlight added

  useEffect(() => {
      // This effect is for highlighting heads based on selectedHeadForHighlight
      const headGroupArraysToUpdate = [
          qHeadsGroupRefs, kHeadsGroupRefs, vHeadsGroupRefs,
          attentionScoresScaledHeadsGroupRefs, attentionWeightsHeadsGroupRefs, headOutputsGroupRefs
      ];
      for (let h = 0; h < numHeads; h++) {
          const intensity = (h === selectedHeadForHighlight) ? SELECTED_HEAD_EMISSIVE_INTENSITY : OTHER_HEAD_EMISSIVE_INTENSITY;
          headGroupArraysToUpdate.forEach(groupArrayRef => {
              if (groupArrayRef.current[h]) {
                  groupArrayRef.current[h].traverse(child => {
                      if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshStandardMaterial) {
                          child.material.emissiveIntensity = intensity;
                          child.material.emissive = new THREE.Color(HIGHLIGHT_COLOR); // Keep emissive color consistent
                      }
                  });
              }
          });
      }
  }, [selectedHeadForHighlight, numHeads, qHeads, kHeads, vHeads, attentionScoresScaled_heads, attentionWeights_heads, headOutputs]);


  useEffect(() => { /* Scene, renderer, camera, static groups, animation loop, tooltip listener */
    // ... (Scene setup as before, no changes here for this subtask part)
    if (!mountRef.current) return; if (!sceneRef.current) sceneRef.current = new THREE.Scene(); sceneRef.current.background = new THREE.Color(0x111119);
    const initialGroups = [ tokenEmbeddingVisGroupRef, peVisGroupRef, finalInputVisGroupRef, wqMhaGroupRef, wkMhaGroupRef, wvMhaGroupRef, qProjectedGroupRef, kProjectedGroupRef, vProjectedGroupRef, concatenatedHeadsGroupRef, woMhaGroupRef, mhaFinalOutputGroupRef, mhaResidualSumGroupRef, mhaNormOutputGroupRef, w1FfnGroupRef, ffnIntermediateLinearGroupRef, ffnReluOutputGroupRef, w2FfnGroupRef, ffnFinalOutputGroupRef, ffnResidualSumGroupRef, ffnNormOutputGroupRef ];
    initialGroups.forEach(groupRef => sceneRef.current?.add(groupRef.current));
    if (!rendererRef.current) {
        rendererRef.current = new THREE.WebGLRenderer({ antialias: true }); rendererRef.current.setSize(window.innerWidth, window.innerHeight); mountRef.current.appendChild(rendererRef.current.domElement);
        cameraRef.current = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000);
        controlsRef.current = new OrbitControls(cameraRef.current, rendererRef.current); controlsRef.current.enableDamping = true;
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.9); sceneRef.current.add(ambientLight); const directionalLight = new THREE.DirectionalLight(0xffffff, 1.3); directionalLight.position.set(10, 25, 15); sceneRef.current.add(directionalLight);
    }
    const onMouseMove = (event: MouseEvent) => { if (!rendererRef.current || !cameraRef.current || !sceneRef.current || !tooltipRef.current) return; const rect = rendererRef.current.domElement.getBoundingClientRect(); mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1; mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1; raycaster.setFromCamera(mouse, cameraRef.current); const interactiveObjects: THREE.Object3D[] = []; allMatrixVisGroups.forEach(groupRef => { if(groupRef.current) interactiveObjects.push(groupRef.current); }); dynamicMatrixGroupArrays.forEach(arrRef => arrRef.current.forEach(group => {if(group) interactiveObjects.push(group);})); const intersects = raycaster.intersectObjects(interactiveObjects, true); if (intersects.length > 0) { const firstIntersect = intersects[0].object; if (firstIntersect.userData && firstIntersect.userData.matrixName) { if (INTERSECTED !== firstIntersect) { tooltipRef.current.style.display = 'block'; tooltipRef.current.innerHTML = `Matrix: ${firstIntersect.userData.matrixName}<br>Pos: [${firstIntersect.userData.row}, ${firstIntersect.userData.col}]<br>Value: ${firstIntersect.userData.value}`; INTERSECTED = firstIntersect; } tooltipRef.current.style.left = `${event.clientX + 10}px`; tooltipRef.current.style.top = `${event.clientY + 10}px`; } else { tooltipRef.current.style.display = 'none'; INTERSECTED = null; } } else { tooltipRef.current.style.display = 'none'; INTERSECTED = null; } };
    rendererRef.current?.domElement.addEventListener('mousemove', onMouseMove);
    if (cameraRef.current && controlsRef.current) { let cMY=MATRIX_BASE_Y_OFFSET+(numHeads*HEAD_VIS_SPACING_Y/2);if(mhaFOGRef.current.children.length>0)cMY=Math.max(cMY,mhaFOGRef.current.position.y);if(mhaNORef.current.children.length>0)cMY=Math.max(cMY,mhaNORef.current.position.y);if(ffnFOGRef.current.children.length>0)cMY=Math.max(cMY,ffnFOGRef.current.position.y);if(ffnNORef.current.children.length>0)cMY=Math.max(cMY,ffnNORef.current.position.y);cameraRef.current.position.set(0,cMY+HEAD_VIS_SPACING_Y,CUBE_SPACING*(tokens.length||2)*3+(numHeads*SDPA_STAGE_SPACING_Z)+(ffnFOGRef.current.children.length>0?HEAD_VIS_SPACING_Y*2:0));controlsRef.current.target.set(0,cMY/2,SDPA_STAGE_SPACING_Z/2);}
    const animate = () => { if (!rendererRef.current || !sceneRef.current || !cameraRef.current || !controlsRef.current) { requestAnimationFrame(animate); return; } requestAnimationFrame(animate); controlsRef.current.update(); rendererRef.current.render(sceneRef.current, cameraRef.current); }; animate();
    const handleResize = () => { if (!cameraRef.current || !rendererRef.current) return; cameraRef.current.aspect = window.innerWidth / window.innerHeight; cameraRef.current.updateProjectionMatrix(); rendererRef.current.setSize(window.innerWidth, window.innerHeight); };
    window.addEventListener('resize', handleResize);
    return () => { window.removeEventListener('resize', handleResize); rendererRef.current?.domElement.removeEventListener('mousemove', onMouseMove); initialGroups.forEach(groupRef => { clearGroup(groupRef.current); sceneRef.current?.remove(groupRef.current); }); [qHeadsGroupRefs, kHeadsGroupRefs, vHeadsGroupRefs, attentionScoresScaledHeadsGroupRefs, attentionWeightsHeadsGroupRefs, headOutputsGroupRefs].forEach(arrRef => { arrRef.current.forEach(g => { clearGroup(g); sceneRef.current?.remove(g); }); arrRef.current = []; });};
  }, [numHeads, tokens.length, dModel, dff]); // Simplified mhaFinalOutputGroupRef to mhaFOGRef etc. for brevity

  return <div ref={mountRef} />;
}

export default App;
