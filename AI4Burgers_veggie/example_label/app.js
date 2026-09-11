const button = document.getElementById('generateButton');
const optimizeGenerateButton = document.getElementById('optimizeGenerateButton');
const personalizeGenerateButton = document.getElementById('personalizeGenerateButton');
const statusEl = document.getElementById('status');
const recipeCard = document.getElementById('recipeCard');
const recipeTitle = document.getElementById('recipeTitle');
const printButton = document.getElementById('printButton');
const calorieBadge = document.getElementById('calorieBadge');
const sustainabilityMetric = document.getElementById('sustainabilityMetric');
const healthMetric = document.getElementById('healthMetric');
const sustainabilityLabel = document.getElementById('sustainabilityLabel');
const healthLabel = document.getElementById('healthLabel');
const sustainabilityFill = document.getElementById('sustainabilityFill');
const healthFill = document.getElementById('healthFill');
const ingredientList = document.getElementById('ingredientList');
const nutritionFacts = document.getElementById('nutritionFacts');
const environmentalImpact = document.getElementById('environmentalImpact');
const sustainabilitySlider = document.getElementById('sustainabilitySlider');
const healthSlider = document.getElementById('healthSlider');
const clickCounter = document.getElementById('clickCounter');
const optimizeTab = document.getElementById('optimizeTab');
const personalizeTab = document.getElementById('personalizeTab');
const optimizePanel = document.getElementById('optimizePanel');
const personalizePanel = document.getElementById('personalizePanel');
const genderMale = document.getElementById('genderMale');
const genderFemale = document.getElementById('genderFemale');
const nnaAge = document.getElementById('nnaAge');
const nnaWeight = document.getElementById('nnaWeight');
const nnaHeightFt = document.getElementById('nnaHeightFt');
const nnaHeightIn = document.getElementById('nnaHeightIn');
const nnaPa = document.getElementById('nnaPa');
const feedbackOpenButton = document.getElementById('feedbackOpenButton');
const feedbackDialog = document.getElementById('feedbackDialog');
const feedbackCloseButton = document.getElementById('feedbackCloseButton');
const feedbackCancelButton = document.getElementById('feedbackCancelButton');
const feedbackForm = document.getElementById('feedbackForm');
const feedbackMessage = document.getElementById('feedbackMessage');
const feedbackContact = document.getElementById('feedbackContact');
const feedbackWebsite = document.getElementById('feedbackWebsite');
const feedbackStatus = document.getElementById('feedbackStatus');
const feedbackSubmitButton = document.getElementById('feedbackSubmitButton');
const GENERATION_MESSAGE = 'BurgerGen is now generating a new burger recipe from 146 ingredients and 10^44 ingredient combinations.';

const defaults = {
  tab: 'optimize',
  sustainability: 0,
  health: 0,
  gender: 'male',
  age: 30,
  weightLb: 176,
  heightFt: 5,
  heightIn: 11,
  activity: 0,
};

let activeTab = defaults.tab;

function setStatus(message, isLoading = false) {
  statusEl.textContent = '';
  statusEl.classList.toggle('is-loading', isLoading);
  if (!message) return;
  statusEl.append(document.createTextNode(message));
  if (isLoading) {
    const spinner = document.createElement('span');
    spinner.className = 'status-spinner';
    spinner.setAttribute('aria-hidden', 'true');
    statusEl.appendChild(spinner);
  }
}

function progressColor(progress) {
  const p = Math.max(0, Math.min(1, Number(progress) || 0));
  const hue = 4 + p * 128;
  return `hsl(${hue}, 66%, 42%)`;
}

function setProgress(fillEl, progress) {
  const p = Math.max(0, Math.min(1, Number(progress) || 0));
  fillEl.style.width = `${Math.round(p * 100)}%`;
  fillEl.style.background = progressColor(p);
}

async function refreshClickCount() {
  try {
    const response = await fetch('/api/stats');
    if (!response.ok) return;
    const stats = await response.json();
    clickCounter.textContent = `# clicks: ${stats.clicks ?? 0}`;
  } catch (error) {
    console.warn('Could not refresh click count', error);
  }
}

refreshClickCount();
setInterval(refreshClickCount, 60 * 60 * 1000);

function resetControls() {
  sustainabilitySlider.value = String(defaults.sustainability);
  healthSlider.value = String(defaults.health);
  genderMale.checked = true;
  genderFemale.checked = false;
  nnaAge.value = String(defaults.age);
  nnaWeight.value = String(defaults.weightLb);
  nnaHeightFt.value = String(defaults.heightFt);
  nnaHeightIn.value = String(defaults.heightIn);
  nnaPa.value = String(defaults.activity);
}

function setActiveTab(tabName) {
  if (tabName === activeTab) return;
  activeTab = tabName;
  resetControls();
  const isOptimize = tabName === 'optimize';
  optimizeTab.classList.toggle('active', isOptimize);
  personalizeTab.classList.toggle('active', !isOptimize);
  optimizeTab.setAttribute('aria-selected', String(isOptimize));
  personalizeTab.setAttribute('aria-selected', String(!isOptimize));
  optimizePanel.classList.toggle('active', isOptimize);
  personalizePanel.classList.toggle('active', !isOptimize);
  optimizePanel.hidden = !isOptimize;
  personalizePanel.hidden = isOptimize;
  setStatus('');
}

function numberValue(input, fallback) {
  const value = Number(input.value);
  return Number.isFinite(value) ? value : fallback;
}

function poundsToKg(pounds) {
  return pounds * 0.45359237;
}

function heightToMeters(feet, inches) {
  return ((feet * 12) + inches) * 0.0254;
}

function activityValue(gender, sliderValue) {
  const progress = Math.max(0, Math.min(1, Number(sliderValue) / 100));
  const maxActivity = gender === 'female' ? 1.48 : 1.45;
  return 1 + progress * (maxActivity - 1);
}

function formatScore(value, digits = 1) {
  const score = Number(value);
  if (!Number.isFinite(score)) return 'unavailable';
  return score.toFixed(digits);
}

function randomPayload() {
  return {
    mode: 'random',
    guidance_scale_sustainability: 0,
    guidance_scale_health: 0,
  };
}

function optimizePayload() {
  return {
    mode: 'optimize',
    guidance_scale_sustainability: Number(sustainabilitySlider.value),
    guidance_scale_health: Number(healthSlider.value),
  };
}

function personalizePayload() {
  const gender = genderFemale.checked ? 'female' : 'male';
  const weightLb = numberValue(nnaWeight, defaults.weightLb);
  const heightFt = numberValue(nnaHeightFt, defaults.heightFt);
  const heightIn = numberValue(nnaHeightIn, defaults.heightIn);
  return {
    mode: 'personalize',
    guidance_scale_sustainability: 0,
    guidance_scale_health: 0,
    nna_gender: gender,
    nna_age: numberValue(nnaAge, defaults.age),
    nna_weight_kg: poundsToKg(weightLb),
    nna_height_m: heightToMeters(heightFt, heightIn),
    nna_pa: activityValue(gender, numberValue(nnaPa, defaults.activity)),
  };
}

optimizeTab.addEventListener('click', () => setActiveTab('optimize'));
personalizeTab.addEventListener('click', () => setActiveTab('personalize'));

function clearRecipe() {
  recipeCard.classList.add('hidden');
  ingredientList.innerHTML = '';
  nutritionFacts.innerHTML = '';
  nutritionFacts.classList.add('hidden');
  environmentalImpact.innerHTML = '';
  environmentalImpact.classList.add('hidden');
  setProgress(sustainabilityFill, 0);
  setProgress(healthFill, 0);
  sustainabilityMetric.removeAttribute('title');
  healthMetric.removeAttribute('title');
  sustainabilityLabel.removeAttribute('title');
  healthLabel.removeAttribute('title');
}

function formatAmount(item) {
  if (item.unavailable || item.amount === null || item.amount === undefined) {
    return '—';
  }
  const amount = Number(item.amount);
  const display = Number.isInteger(amount) ? String(amount) : String(amount.toFixed(1)).replace(/\\.0$/, '');
  return `${display}${item.unit || ''}`;
}

function appendNutritionRow(parent, item, className = '') {
  const row = document.createElement('div');
  row.className = `nutrition-row ${item.indent ? 'indent' : ''} ${className}`.trim();

  const label = document.createElement('div');
  label.className = 'nutrition-label';
  label.textContent = item.label;

  const amount = document.createElement('span');
  amount.className = 'nutrition-amount';
  amount.textContent = ` ${formatAmount(item)}`;
  label.appendChild(amount);

  const dv = document.createElement('div');
  dv.className = 'nutrition-dv';
  dv.textContent = item.dv_percent === null || item.dv_percent === undefined ? '' : `${item.dv_percent}%`;

  row.append(label, dv);
  parent.appendChild(row);
}

function appendFactsHeader(parent, { title, servingSize = '1 burger', calories = 500, showDailyValueHeader = false }) {
  const titleElement = document.createElement('div');
  titleElement.className = 'nutrition-title';
  titleElement.textContent = title;

  const serving = document.createElement('div');
  serving.className = 'nutrition-serving';
  serving.textContent = `Serving size ${servingSize}`;

  const caloriesElement = document.createElement('div');
  caloriesElement.className = 'nutrition-calories';
  caloriesElement.innerHTML = `<span>Calories</span><strong>${Math.round(calories || 500)}</strong>`;

  parent.append(titleElement, serving, caloriesElement);

  if (showDailyValueHeader) {
    const dvHeader = document.createElement('div');
    dvHeader.className = 'nutrition-dv-header';
    dvHeader.textContent = '% Daily Value*';
    parent.appendChild(dvHeader);
  }
}

function appendFactsValueRow(parent, labelText, valueNode, className = '') {
  const row = document.createElement('div');
  row.className = `nutrition-row ${className}`.trim();

  const label = document.createElement('div');
  label.className = 'nutrition-label';
  label.textContent = labelText;

  const value = document.createElement('div');
  value.className = `nutrition-dv ${className ? `${className}-value` : ''}`.trim();
  if (valueNode instanceof Node) {
    value.appendChild(valueNode);
  } else {
    value.textContent = String(valueNode ?? '');
  }

  row.append(label, value);
  parent.appendChild(row);
}

function renderNutritionFacts(facts) {
  nutritionFacts.innerHTML = '';
  if (!facts) {
    nutritionFacts.classList.add('hidden');
    return;
  }

  appendFactsHeader(nutritionFacts, {
    title: 'Nutrition Facts',
    servingSize: facts.serving_size || '1 burger',
    calories: facts.calories || 500,
    showDailyValueHeader: true,
  });

  for (const item of facts.items || []) {
    appendNutritionRow(nutritionFacts, item);
  }

  const micronutrientDivider = document.createElement('div');
  micronutrientDivider.className = 'nutrition-divider-thin';
  nutritionFacts.appendChild(micronutrientDivider);

  for (const item of facts.micronutrients || []) {
    appendNutritionRow(nutritionFacts, item, 'micronutrient');
  }

  const note = document.createElement('p');
  note.className = 'nutrition-note';
  note.textContent = facts.daily_value_note || '';
  nutritionFacts.appendChild(note);
  nutritionFacts.classList.remove('hidden');
}

function formatEnvironmentalAmount(item) {
  const amount = Number(item.amount);
  if (!Number.isFinite(amount)) return '—';
  const decimals = Number.isInteger(item.decimals) ? item.decimals : 2;
  const display = amount.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return display;
}

function createEnvironmentalValue(item) {
  const value = document.createElement('span');
  value.textContent = formatEnvironmentalAmount(item);

  const unit = item.unit;
  const unitSpan = document.createElement('span');
  unitSpan.className = 'environmental-unit';

  if (unit === 'm^2') {
    unitSpan.append(document.createTextNode('m'));
    const sup = document.createElement('sup');
    sup.textContent = '2';
    unitSpan.appendChild(sup);
  } else if (unit === 'kg PO4^3- eq') {
    unitSpan.append(document.createTextNode('kg PO'));
    const sub = document.createElement('sub');
    sub.textContent = '4';
    unitSpan.appendChild(sub);
    const sup = document.createElement('sup');
    sup.textContent = '3-';
    unitSpan.appendChild(sup);
    unitSpan.append(document.createTextNode(' eq'));
  } else {
    unitSpan.textContent = unit || '';
  }

  if (unitSpan.textContent) {
    value.append(document.createTextNode(' '), unitSpan);
  }
  return value;
}

function orderedEnvironmentalItems(items) {
  const order = [
    'Land Use',
    'Eutrophication Potential',
    'Water Use',
    'Greenhouse Gas Emissions',
  ];
  const orderedLabels = new Set(order);
  const byLabel = new Map(items.map((item) => [item.label, item]));
  return [
    ...order.map((label) => byLabel.get(label)).filter(Boolean),
    ...items.filter((item) => !orderedLabels.has(item.label)),
  ];
}

function renderEnvironmentalImpact(categories) {
  environmentalImpact.innerHTML = '';
  if (!categories || !Array.isArray(categories.items)) {
    environmentalImpact.classList.add('hidden');
    return;
  }

  appendFactsHeader(environmentalImpact, {
    title: 'Sustainability Facts',
    servingSize: '1 burger',
    calories: 500,
  });

  for (const item of orderedEnvironmentalItems(categories.items)) {
    appendFactsValueRow(environmentalImpact, item.label, createEnvironmentalValue(item), 'sustainability-row');
  }

  environmentalImpact.classList.remove('hidden');
}

function renderRecipe(recipe) {
  recipeTitle.textContent = recipe.title || 'AI-Generated Burger Recipe';
  calorieBadge.textContent = `${Math.round(recipe.serving_calories || 500)} kcal`;
  setProgress(sustainabilityFill, recipe.metrics?.sustainability?.progress);
  setProgress(healthFill, recipe.metrics?.health?.progress);
  const sustainabilityTitle = `Sustainability score: ${formatScore(recipe.metrics?.sustainability?.score, 2)}`;
  const healthTitle = `Healthy Eating Index: ${formatScore(recipe.metrics?.health?.score, 1)}`;
  sustainabilityMetric.title = sustainabilityTitle;
  sustainabilityLabel.title = sustainabilityTitle;
  healthMetric.title = healthTitle;
  healthLabel.title = healthTitle;
  ingredientList.innerHTML = '';

  for (const ingredient of recipe.ingredients || []) {
    const row = document.createElement('div');
    row.className = 'ingredient-row';

    const name = document.createElement('div');
    name.className = 'ingredient-name';
    name.textContent = ingredient.name;

    const grams = document.createElement('div');
    grams.className = 'ingredient-grams';
    grams.textContent = `${ingredient.grams} g`;

    row.append(name, grams);
    ingredientList.appendChild(row);
  }

  renderNutritionFacts(recipe.nutrition_facts);
  renderEnvironmentalImpact(recipe.environmental_impact_categories);
  recipeCard.classList.remove('hidden');
}

function setGenerating(isGenerating) {
  button.disabled = isGenerating;
  optimizeGenerateButton.disabled = isGenerating;
  personalizeGenerateButton.disabled = isGenerating;
}

async function generateRecipe(payload, loadingMessage) {
  setGenerating(true);
  clearRecipe();
  setStatus(loadingMessage, true);

  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      throw new Error(`Generation failed with status ${response.status}`);
    }
    const recipe = await response.json();
    renderRecipe(recipe);
    if (recipe.clicks !== undefined) {
      clickCounter.textContent = `# clicks: ${recipe.clicks}`;
    } else {
      refreshClickCount();
    }
    setStatus(loadingMessage);
  } catch (error) {
    console.error(error);
    setStatus('Something went wrong while generating the burger. Please try again.');
  } finally {
    setGenerating(false);
  }
}

printButton.addEventListener('click', () => window.print());

function openFeedbackDialog() {
  feedbackDialog.classList.remove('hidden');
  feedbackStatus.textContent = '';
  feedbackMessage.focus();
}

function closeFeedbackDialog() {
  feedbackDialog.classList.add('hidden');
  feedbackForm.reset();
  feedbackStatus.textContent = '';
}

feedbackOpenButton.addEventListener('click', openFeedbackDialog);
feedbackCloseButton.addEventListener('click', closeFeedbackDialog);
feedbackCancelButton.addEventListener('click', closeFeedbackDialog);

feedbackDialog.addEventListener('click', (event) => {
  if (event.target === feedbackDialog) {
    closeFeedbackDialog();
  }
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !feedbackDialog.classList.contains('hidden')) {
    closeFeedbackDialog();
  }
});

feedbackForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  feedbackSubmitButton.disabled = true;
  feedbackStatus.textContent = 'Sending feedback...';

  try {
    const response = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: feedbackMessage.value,
        contact: feedbackContact.value,
        website: feedbackWebsite.value,
        page_url: window.location.href,
      }),
    });
    if (!response.ok) {
      throw new Error(`Feedback failed with status ${response.status}`);
    }
    feedbackStatus.textContent = 'Thank you. Your feedback was sent.';
    feedbackForm.reset();
    window.setTimeout(closeFeedbackDialog, 1000);
  } catch (error) {
    console.error(error);
    feedbackStatus.textContent = 'Sorry, feedback could not be sent. Please try again.';
  } finally {
    feedbackSubmitButton.disabled = false;
  }
});

button.addEventListener('click', () => {
  generateRecipe(randomPayload(), GENERATION_MESSAGE);
});

optimizeGenerateButton.addEventListener('click', () => {
  generateRecipe(optimizePayload(), GENERATION_MESSAGE);
});

personalizeGenerateButton.addEventListener('click', () => {
  generateRecipe(personalizePayload(), GENERATION_MESSAGE);
});
