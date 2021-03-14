package jPBC_3;

import java.io.File;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import it.unisa.dia.gas.jpbc.Element;
import it.unisa.dia.gas.jpbc.Pairing;
import it.unisa.dia.gas.plaf.jpbc.pairing.PairingFactory;
import jPBC_2.ChuShiHua;

public class Client_init {
	public Pairing pairing;
	public Element u,g,x,y;
	public Element TiElement;
	public Element T;
	public Element P;
	public Element tElement;
	public Element pElement;
	public Element leftResultElement,rightResultElement,rightResultHF;
	public List<Element> TiListElement;
	public List<Element> HFiListElement;
	public List<Element> HFiListElementright;
	public List<Element> UFiListElement;
	public HashMap<Integer,Element> ViMapElement;
	long l;
	public void init() {
		this.pairing = PairingFactory.getPairing("a.properties");
		this.u = pairing.getG1().newRandomElement().getImmutable();
		this.g = pairing.getG1().newRandomElement().getImmutable();
		this.x = pairing.getZr().newRandomElement().getImmutable();
		this.y = this.g.powZn(this.x).getImmutable();
		System.out.println("初始化成功");
		System.out.println("---------------");
		
	}
	public void creatTi(){
		List<String> fileMD5list = Utils.mD5Util(1, 3);
		List<String> fileMD5listright = Utils.mD5Utilcloud(1, 3);
		TiElement =pairing.getG1().newOneElement();
		//HFiElement = pairing.getG1().newOneElement();
		//uFiElement = pairing.getG1().newOneElement();
		T = pairing.getG1().newOneElement();//
		P = pairing.getZr().newZeroElement();// 
		rightResultHF = pairing.getG1().newOneElement();
		TiListElement = new ArrayList<>();
		HFiListElement = new ArrayList<>();
		HFiListElementright = new ArrayList<>();
		UFiListElement = new ArrayList<>();
		ViMapElement = new HashMap<>();
		System.out.println("T:"+T);
		System.out.println("P:"+P);
		for (int i = 1; i <= fileMD5list.size(); i++) {
			Element HFiElement = pairing.getG1().newOneElement().setFromHash(fileMD5list.get(i-1)
					.getBytes(), 0, fileMD5list.get(i-1).length()).getImmutable();
			HFiListElement.add(HFiElement.duplicate());
			l = (int)i;
			Element uFiElement=this.u.pow(BigInteger.valueOf(l)).getImmutable();
			TiElement = (HFiElement.mul(uFiElement).getImmutable()).powZn(this.x).getImmutable();
			System.out.println("第"+i+"次TiElement:"+TiElement);
			TiListElement.add(TiElement.duplicate());
			UFiListElement.add(uFiElement.duplicate());			
		}
		/*		
			计算右边的HFi
			*/
		for(int i = 1;i<=fileMD5listright.size();i++){
			Element HFiElement = pairing.getG1().newOneElement().setFromHash(fileMD5listright.get(i-1).getBytes(), 0, fileMD5listright.get(i-1).length()).getImmutable();
			HFiListElementright.add(HFiElement.getImmutable());
		}
		for (int i = 1; i <= TiListElement.size(); i++) {
			l = (int)i;
			Element Vi = pairing.getZr().newRandomElement().getImmutable();
			tElement = TiListElement.get(i-1).powZn(Vi);
			T.mul(tElement).getImmutable();
			pElement = Vi.mul(BigInteger.valueOf(l));
			P.add(pElement);
			System.out.println("第"+(i)+"回T的值"+T);
			System.out.println("第"+(i)+"回P的值"+P);
			ViMapElement.put(i, Vi.getImmutable());
			
			rightResultHF.mul(HFiListElementright.get(i-1).powZn(Vi));
		}
		T.getImmutable();
		P.getImmutable();
	    leftResultElement = pairing.pairing(T, g);
	    rightResultElement = pairing.pairing(rightResultHF.mul(this.u.powZn(P)), this.y);
	    System.out.println("leftResultElement:"+leftResultElement);
		System.out.println("rightResultElement"+rightResultElement);
		System.out.println(leftResultElement.equals(rightResultElement));
	}
}
